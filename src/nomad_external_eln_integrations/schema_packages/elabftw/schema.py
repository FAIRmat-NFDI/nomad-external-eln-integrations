#
# Copyright The NOMAD Authors.
#
# This file is part of NOMAD. See https://nomad-lab.eu for further info.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
import json
import math
import os
import re
from urllib.parse import parse_qs, urlparse

import jmespath
import numpy as np
import requests
import yaml
from nomad.config import config
from nomad.datamodel.data import ArchiveSection
from nomad.metainfo import (
    MSection,
    Quantity,
    Section,
    SubSection,
)
from nomad.metainfo.metainfo import Property, SchemaPackage
from nomad.units import ureg


m_package = SchemaPackage()


class ElabftwImportError(Exception):
    """elabftw-normalizer related errors"""

    pass


class ElabftwProject(ArchiveSection):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.__headers = None
        self.logger = None

    project_url = Quantity(type=str, a_eln=dict(component='StringEditQuantity'))
    api_url = Quantity(type=str, a_eln=dict(component='StringEditQuantity'))
    api_key = Quantity(
        type=str,
        a_eln=dict(component='StringEditQuantity', props=dict(type='password')),
    )
    Sync_Project = Quantity(type=bool, a_eln=dict(component='BoolEditQuantity'))

    def _elab_api_method(
        self,
        method,
        api_base_url,
        target_endpoint,
        msg='cannot do elabftw api request',
        **kwargs,
    ):
        response = method(
            f'{api_base_url}{target_endpoint}',
            headers=self._headers,
            timeout=10,
            **kwargs,
        )

        if response.status_code >= 400:
            self.logger.error(
                msg, data=dict(status_code=response.status_code, text=response.text)
            )
            raise ElabftwImportError()

        return response

    def _extract_url_components(self):
        try:
            parsed = urlparse(self.project_url)
            base_url = f'{parsed.scheme}://{parsed.netloc}'
            api_base_url = os.path.join(base_url, 'api', 'v2')
            queries = {k: v[0] for k, v in parse_qs(parsed.query).items()}

            experiment_id_str = queries.get('id')
            try:
                exp_id = int(experiment_id_str) if experiment_id_str else None
            except ValueError:
                exp_id = None

            return api_base_url, queries, exp_id
        except Exception:
            raise ElabftwImportError(
                'Could not parse the url. Make sure the url is correct.'
            )

    @property
    def _headers(self):
        if not self.__headers:
            self.__headers = dict(Authorization=f'{self.api_key}')

        return self.__headers

    def _clear_user_data(self):
        self.api_key = None

        archive = self.m_root()
        with archive.m_context.raw_file(archive.metadata.mainfile, 'wt') as f:
            if archive.metadata.mainfile.endswith('json'):
                json.dump(dict(data=archive.data.m_to_dict()), f)
            else:
                yaml.dump(dict(data=archive.data.m_to_dict()), f)

    def normalize(self, archive, logger):
        super().normalize(archive, logger)
        self.logger = logger

        if self.Sync_Project:
            if not self.project_url or not self.api_key:
                logger.error('missing information, cannot import project')
                raise ElabftwImportError()

            try:
                api_base_url, query_params, experiment_id = (
                    self._extract_url_components()
                )

            except KeyError as e:
                logger.error('cannot parse project id from url', exc_info=e)
                raise ElabftwImportError()
            if self.api_url:
                api_base_url = self.api_url

            experiment = self._elab_api_method(
                requests.get,
                api_base_url,
                f'/experiments/{experiment_id}?format=json&json=true&withTitle=true',
            ).json()

            data = _remove_spaces_from_keys(experiment)

            mappings = _create_column_to_quantity_mapping(
                self.m_def, annotation_hook='elabftw'
            )
            for jmes_path, mapping in mappings:
                value = _extract_data(data, jmes_path)
                mapping(self, value)

        self.Sync_Project = False
        logger.info('Parsing finished.')


m_package.__init_metainfo()


def _extract_data(data, jmes_path):
    """
    Extracts data from a given data structure using a jmespath expression.

    Parameters:
    - data (dict): The data to extract from.
    - jmes_path (str): The jmespath expression.

    Returns:
    - The extracted data or None if not found.
    """
    try:
        result = jmespath.search(_clean_jmespath_expression(jmes_path), data)
        return result
    except jmespath.exceptions.JMESPathError as e:
        print(f"Error parsing jmespath '{jmes_path}': {e}")
        return None


def _create_column_to_quantity_mapping(section_def: Section, annotation_hook: str):
    """
    Creates a mapping as a set of tuples, where each tuple contains a jmes_path string and a corresponding set_value function.

    Parameters:
    - section_def (Section): The section definition to process.
    - annotation_hook (str): The annotation hook to use.

    Returns:
    - mappings: A set of (jmes_path, set_value) tuples.
    """
    mappings: set[tuple[str, callable[[MSection, any], None]]] = set()

    def add_section_def(section_def: Section, path: list[tuple[SubSection, Section]]):
        properties: set[Property] = set()

        for quantity in section_def.all_quantities.values():
            annotation = quantity.m_get_annotations(annotation_hook)
            annotation = annotation[0] if isinstance(annotation, list) else annotation
            if annotation:
                jmes_path = annotation.get('path', None)

                def set_value(
                    section: MSection,
                    value,
                    path=path,
                    quantity=quantity,
                ):
                    for sub_section, path_section_def in path:
                        next_section = None
                        try:
                            next_section = section.m_get_sub_section(sub_section, -1)
                        except (KeyError, IndexError):
                            pass
                        if not next_section:
                            next_section = path_section_def.section_cls()
                            section.m_add_sub_section(sub_section, next_section, -1)
                        section = next_section

                    if annotation:
                        if unit := annotation.get('unit', None):
                            value *= ureg(unit)

                    if isinstance(value, float) and math.isnan(value):
                        value = None

                    if isinstance(value, (int, float, str)):
                        value = np.array([value])

                    if value is not None:
                        if len(value.shape) == 1 and len(quantity.shape) == 0:
                            if len(value) == 1:
                                value = value[0]
                            elif len(value) == 0:
                                value = None
                            else:
                                raise ElabftwImportError(
                                    f'The shape of {quantity.name} does not match the given data.'
                                )
                        elif len(value.shape) != len(quantity.shape):
                            raise ElabftwImportError(
                                f'The shape of {quantity.name} does not match the given data.'
                            )

                    section.m_set(quantity, value)

                mappings.add((jmes_path, set_value))

        for sub_section in section_def.all_sub_sections.values():
            if sub_section in properties:
                continue
            next_base_section = sub_section.sub_section
            properties.add(sub_section)
            add_section_def(
                next_base_section,
                path
                + [
                    (
                        sub_section,
                        next_base_section,
                    )
                ],
            )

    add_section_def(section_def, [])
    return mappings


def _remove_spaces_from_keys(data):
    """
    Recursively removes white spaces from the keys of a nested dictionary.

    Parameters:
    - data: The input data structure (dict, list, or other).

    Returns:
    - A new dictionary/list with white spaces removed from dictionary keys.
    """
    if isinstance(data, dict):
        new_dict = {}
        for key, value in data.items():
            if isinstance(key, str):
                new_key = key.replace(' ', '')
            else:
                new_key = key
            new_dict[new_key] = _remove_spaces_from_keys(value)
        return new_dict
    elif isinstance(data, list):
        return [_remove_spaces_from_keys(item) for item in data]
    else:
        return data


def _clean_jmespath_expression(expr: str) -> str:
    """
    Removes white spaces from the keys in a JMESPath-like expression. The Elabftw api responses may contain keys with
    white spaces which is troublesome for parsing values using jmespath library.

    Args:
        expr (str): The original JMESPath expression with spaces in keys.

    Returns:
        str: The cleaned JMESPath expression with spaces removed from keys.
    """
    pattern = re.compile(r'\.(?![^\[]*\])')

    parts = pattern.split(expr)

    cleaned_parts = []
    for part in parts:
        match = re.match(r'^([^\[]+)(\[\d+\])?$', part)
        if match:
            key = match.group(1)
            index = match.group(2) if match.group(2) else ''
            cleaned_key = key.replace(' ', '')
            cleaned_part = f'{cleaned_key}{index}'
            cleaned_parts.append(cleaned_part)
        else:
            cleaned_parts.append(part)

    cleaned_expr = '.'.join(cleaned_parts)
    return cleaned_expr
