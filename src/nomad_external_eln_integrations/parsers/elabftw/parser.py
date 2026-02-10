#
# Copyright The NOMAD Authors.
#
# This file is part of NOMAD.
# See https://nomad-lab.eu for further info.
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
import copy
import json
import os
import re
from collections.abc import Iterable
from typing import Union

from nomad import utils
from nomad.datamodel import ArchiveSection, EntryArchive, EntryData, Results
from nomad.datamodel.data import ElnIntegrationCategory
from nomad.datamodel.metainfo.annotations import ELNAnnotation
from nomad.metainfo import JSON, Datetime, MSection, Quantity, Section, SubSection
from nomad.metainfo.util import MEnum, camel_case_to_snake_case
from nomad.parsing import MatchingParser


def _remove_at_sign_from_keys(obj):
    obj = copy.deepcopy(obj)

    for k, v in list(obj.items()):
        if k.startswith('@'):
            obj[k.lstrip('@')] = v
            del obj[k]
            k = k.lstrip('@')
        if isinstance(v, dict):
            obj[k] = _remove_at_sign_from_keys(v)
        if isinstance(v, list):
            for i, item in enumerate(v):
                if isinstance(item, dict):
                    obj[k][i] = _remove_at_sign_from_keys(item)

    return obj


def _map_response_to_dict(data: list) -> dict:
    mapped_dict: dict = {}
    cleaned_data = _remove_at_sign_from_keys(data)
    for item in cleaned_data['graph']:
        id = item['id']
        if id not in mapped_dict:
            mapped_dict[id] = item
        else:
            num = int(id.split('__internal_')[1])
            id_internal = f'{id}__internal_{num + 1}'
            mapped_dict[id_internal] = item
    return mapped_dict


def _create_file_section(file, graph, parent_folder_raw_path, logger=None):
    try:
        section = _element_type_section_mapping[file['type']]()
    except Exception:
        logger.error('Could not find type for the file')
        raise ELabFTWParserError(f'Could not find type fo the file {file["id"]}')
    tmp = {k: v[0] if isinstance(v, list) else v for k, v in graph[file['id']].items()}
    section.m_update_from_dict(tmp)
    try:
        file_name = file['id'].split('./')[1]
        full_path = os.path.join(parent_folder_raw_path, file_name)
        section.post_process(file_name=full_path)
    except Exception:
        logger.error('Could not set the file path for file')
    return section


status_types = MEnum(
    'Not set',
    'Running',
    'Waiting',
    'Success',
    'Need to be redone',
    'Fail',
    'Maintenance mode',
    'Need to reorder',
    'Processed',
)


class ELabFTWRef(MSection):
    """Represents a referenced item in ELabFTW entry."""

    row_refs = Quantity(
        type=ArchiveSection,
        a_eln=ELNAnnotation(
            component='ReferenceEditQuantity',
            label='ELabFTW Reference',
        ),
        description='References that connect to each ELabFTW ref. Each item is stored in it individual entry.',
    )


class ELabFTWParserError(Exception):
    """eln parser related errors."""

    pass


class ELabFTWExperimentLink(MSection):
    """
    This class contains information from other experiments that are linked to this specific experiment.
    the external link can be accessed using the query parameter #id:
    https://demo.elabftw.net/experiments.php?mode=view&id={itemid}
    """

    itemid = Quantity(
        type=str, description='id of the external experiment linked to this experiment'
    )
    id = Quantity(
        type=str, description='id of the experiment linked to this experiment'
    )
    title = Quantity(type=str, description='title of the external experiment')
    elabid = Quantity(type=str, description='hashed id')
    category = Quantity(
        type=str, description='Category/status of the external experiment link'
    )


class ELabFTWItemLink(ELabFTWExperimentLink):
    """
    This class holds information of the items related to this specific experiment.
    The external link can be accessed via setting the query parameter #related:
    https://demo.elabftw.net/database.php?mode=show&related={itemid}
    """

    bookable = Quantity(type=bool)


class ELabFTWSteps(MSection):
    """
    Steps recorded for the current experiment
    """

    id = Quantity(type=str, description='id of the current step')
    item_id = Quantity(type=str, description='item_id of the current experiment')
    body = Quantity(type=str, description='title of the step')
    ordering = Quantity(
        type=str, description='location of the current step in the overall order'
    )
    finished = Quantity(
        type=bool, description='a boolean if the step is taken/finished'
    )
    finished_time = Quantity(
        type=Datetime, description='time at which the step is finished'
    )
    deadline = Quantity(type=Datetime, description='deadline time')


class ELabFTWExperimentData(MSection):
    """
    Detailed information of the given ELabFTW experiment, such as links to external resources and extra fields, are
    stored here.
    """

    # Enable this section on the Overview page
    m_def = Section(a_eln=ELNAnnotation(overview=True, label='Experiment Data'))

    type = Quantity(
        type=str,
        description='Type of the experiment (e.g. from eLabFTW template)',
        a_eln=ELNAnnotation(component='StringEditQuantity'),
    )

    body = Quantity(
        type=str,
        description='an html-tagged string containing the information of this experiment',
        # Configure the Rich Text Editor and show on Overview
        a_eln=ELNAnnotation(
            component='RichTextEditQuantity', overview=True, label='Description'
        ),
        a_browser=dict(render_value='HtmlValue'),
    )

    created_at = Quantity(
        type=Datetime,
        description='Date and time of when this experiment is created at.',
    )
    sharelink = Quantity(
        type=str,
        a_eln=dict(component='URLEditQuantity'),
        description='URL link to this experiment in the ELabFTW repository',
    )
    extra_fields = Quantity(
        type=JSON,
        description='data in the extra_fields field',
        a_browser=dict(value_component='JsonValue'),
    )
    firstname = Quantity(
        type=str,
        a_eln=dict(component='StringEditQuantity'),
        description="Author's first name",
    )
    fullname = Quantity(
        type=str,
        a_eln=dict(component='StringEditQuantity'),
        description="Author's full name",
    )
    tags = Quantity(
        type=str,
        shape=['*'],
        a_eln=dict(component='StringEditQuantity'),
        description='Tags',
    )

    items_links = SubSection(sub_section=ELabFTWItemLink, repeats=True)
    experiments_links = SubSection(sub_section=ELabFTWExperimentLink, repeats=True)
    steps = SubSection(sub_section=ELabFTWSteps, repeats=True)
    references = SubSection(sub_section=ELabFTWRef, repeats=True)

    def normalize(self, archive, logger) -> None:
        exp_ids = [('experiments', exp.itemid) for exp in self.experiments_links]
        res_ids = [('database', exp.itemid) for exp in self.items_links]
        try:
            for item in exp_ids + res_ids:
                from nomad.search import MetadataPagination, search

                query = {'external_id': item[1]}
                search_result = search(
                    owner='all',
                    query=query,
                    pagination=MetadataPagination(page_size=1),
                    user_id=archive.metadata.main_author.user_id,
                )
                if search_result.pagination.total > 0:
                    entry_id = search_result.data[0]['entry_id']
                    upload_id = search_result.data[0]['upload_id']
                    ref = ELabFTWRef()
                    ref.row_refs = f'../uploads/{upload_id}/archive/{entry_id}#data'
                    self.references.append(ref)
                    if search_result.pagination.total > 1:
                        logger.warn(
                            'Found multiple entries with external id. Will use the first one found.'
                        )
                else:
                    logger.warn('Found no entries with metadata.external_id')
        except Exception:
            logger.warning(
                'Could not fetch referenced experiments internally. '
                'This is normal if the referenced experiments are not in the same upload.'
            )


class ELabFTWComment(MSection):
    """
    A section containing comments made on the experiment. It contains a user object that refers to the id of the
    comment creator
    """

    date_created = Quantity(type=Datetime, description='Creation date of the comment')
    text = Quantity(type=str, description="Comment's content")
    author = Quantity(
        type=JSON,
        description='author information',
        a_browser=dict(value_component='JsonValue'),
    )


class ELabFTWBaseSection(MSection):
    """
    General information on the exported files/experiment of the .eln file
    """

    m_def = Section(label_quantity='type')

    id = Quantity(type=str, description='id of the current data-type')
    type = Quantity(type=str, description='type of the data')

    def post_process(self, **kwargs):
        pass


class ELabFTWFile(ELabFTWBaseSection):
    """
    Information of the exported files
    """

    description = Quantity(type=str, description='Description of the file')
    name = Quantity(type=str, description='Name of the file')
    content_size = Quantity(type=str, description='Size of the file')
    content_type = Quantity(type=str, description='Type of this file')
    file = Quantity(type=str, a_browser=dict(adaptor='RawFileAdaptor'))

    def post_process(self, **kwargs):
        file_name = kwargs.get('file_name', None)
        self.file = file_name


class ElabFTWDataset(ELabFTWBaseSection):
    """
    Information of the dataset type. The author information goes here.
    """

    author = Quantity(
        type=JSON,
        description='author information',
        a_browser=dict(value_component='JsonValue'),
    )
    name = Quantity(type=str, description='Name of the Dataset')
    text = Quantity(type=str, description='Body content of the dataset')
    url = Quantity(
        type=str,
        a_eln=dict(component='URLEditQuantity'),
        description='Link to this dataset in ELabFTW repository',
    )
    date_created = Quantity(type=Datetime, description='Creation date')
    date_modified = Quantity(type=Datetime, description='Last modification date')
    keywords = Quantity(
        type=str,
        shape=['*'],
        description='keywords associated with the current experiment',
    )

    comment = SubSection(sub_section=ELabFTWComment, repeats=True)


class ELabFTW(EntryData):
    """
    Each exported .eln formatted file contains ro-crate-metadata.json file which is parsed into this class.
    Important Quantities are:
        id: id of the file which holds metadata info
        date_created: date of when the file is exported

    title is used as an identifier for the GUI to differentiate between the parsed entries and the original file.
    """

    m_def = Section(
        label='ELabFTW Project Import',
        categories=[ElnIntegrationCategory],
        a_eln=ELNAnnotation(overview=True),
    )

    id = Quantity(
        type=str,
        description='id of the file containing the metadata information. It should always be ro-crate-metadata.json',
    )
    title = Quantity(type=str, description='Title of the entry')

    date_created = Quantity(type=Datetime, description='Creation date of the .eln')
    sd_publisher = Quantity(
        type=JSON,
        description='Publisher information',
        a_browser=dict(value_component='JsonValue'),
    )

    author = Quantity(type=str, description="Full name of the experiment's author")
    project_id = Quantity(type=str, description='Project ID')
    status = Quantity(
        type=status_types,
        description='Status of the Experiment',
    )
    keywords = Quantity(
        type=str, shape=['*'], description='Keywords associated with the Experiment'
    )

    experiment_data = SubSection(
        sub_section=ELabFTWExperimentData, a_eln=ELNAnnotation(overview=True)
    )
    experiment_files = SubSection(sub_section=ELabFTWBaseSection, repeats=True)

    def post_process(self, **kwargs):
        full_name = kwargs.get('full_name')
        self.author = full_name


_element_type_section_mapping = {'File': ELabFTWFile, 'Dataset': ElabFTWDataset}


class ELabFTWParser(MatchingParser):
    creates_children = True

    def is_mainfile(
        self,
        filename: str,
        mime: str,
        buffer: bytes,
        decoded_buffer: str,
        compression: str = None,
    ) -> Union[bool, Iterable[str]]:
        is_ro_crate = super().is_mainfile(
            filename, mime, buffer, decoded_buffer, compression
        )
        if not is_ro_crate:
            return False
        try:
            with open(filename) as f:
                data = json.load(f)
        except Exception:
            return False

        try:
            if any(
                item.get('@type') == 'SoftwareApplication' for item in data['@graph']
            ):
                root_experiment = next(
                    item for item in data['@graph'] if item.get('@id') == './'
                )
                no_of_experiments = len(root_experiment['hasPart'])
            else:
                no_of_experiments = len(
                    [item['hasPart'] for item in data['@graph'] if item['@id'] == './'][
                        0
                    ]
                )
        except (KeyError, IndexError, TypeError):
            return False

        return [str(item) for item in range(0, no_of_experiments)]

    def parse(
        self, mainfile: str, archive: EntryArchive, logger=None, child_archives=None
    ):
        if logger is None:
            logger = utils.get_logger(__name__)

        title_pattern = re.compile(r'^\d{4}-\d{2}-\d{2} - ([a-zA-Z0-9\-]+) - .*$')

        lab_ids: list[tuple[str, str]] = []
        with open(mainfile) as f:
            data = json.load(f)

        snake_case_data = camel_case_to_snake_case(data)
        clean_data = _remove_at_sign_from_keys(snake_case_data)
        graph = {item['id']: item for item in clean_data['graph']}
        experiments = graph['./']

        for index, experiment in enumerate(experiments['has_part']):
            exp_id = experiment['id']
            raw_experiment, exp_archive = graph[exp_id], child_archives[str(index)]

            # hook for matching the older .eln files from Elabftw exported files
            if not any(
                item.get('type') == 'SoftwareApplication'
                for item in clean_data['graph']
            ):
                elabftw_experiment = _parse_legacy(
                    graph,
                    raw_experiment,
                    exp_archive,
                    data,
                    mainfile,
                    exp_id,
                    title_pattern,
                    lab_ids,
                    archive,
                    logger,
                )

                if not archive.results:
                    archive.results = Results()
                    archive.results.eln = Results.eln.sub_section.section_cls()
                    archive.results.eln.lab_ids = [str(lab_id[1]) for lab_id in lab_ids]
                    archive.results.eln.tags = [lab_id[0] for lab_id in lab_ids]

            else:
                elabftw_experiment = _parse_latest(
                    graph,
                    raw_experiment,
                    exp_archive,
                    mainfile,
                    exp_id,
                    archive,
                    logger,
                )

            exp_archive.data = elabftw_experiment

        logger.info('eln parsed successfully')


def _parse_legacy(
    graph,
    raw_experiment,
    exp_archive,
    data,
    mainfile,
    exp_id,
    title_pattern,
    lab_ids,
    archive,
    logger,
) -> ELabFTW:
    elabftw_experiment = ELabFTW()
    try:
        del graph['ro-crate-metadata.json']['type']
    except Exception:
        pass
    tmp = {
        k: v[0] if isinstance(v, list) else v
        for k, v in graph['ro-crate-metadata.json'].items()
    }
    elabftw_experiment.m_update_from_dict(tmp)
    elabftw_entity_type = _set_experiment_metadata(
        raw_experiment, exp_archive, elabftw_experiment, logger
    )

    # Extract author with backward compatibility
    author_full_name = None
    try:
        # Strategy 1: Search for Person or Author type in the graph
        author_types = ['Person', 'Author', 'schema:Person', 'schema:Author']
        author_item = next(
            (item for item in data['@graph'] if item.get('@type') in author_types), None
        )

        if author_item:
            # Try different key variants for first/last name
            given_name = (
                author_item.get('givenName')
                or author_item.get('given_name')
                or author_item.get('firstName')
                or ''
            )
            family_name = (
                author_item.get('familyName')
                or author_item.get('family_name')
                or author_item.get('lastName')
                or ''
            )
            # Try full name if available
            full_name = author_item.get('name') or author_item.get('fullName')

            if full_name:
                author_full_name = full_name
            elif given_name or family_name:
                author_full_name = f'{given_name} {family_name}'.strip()

        # Strategy 2: Fallback to legacy approach (last item in graph with name fields)
        if not author_full_name:
            last_item = data['@graph'][-1] if data.get('@graph') else None
            if last_item:
                given = last_item.get('given_name') or last_item.get('givenName') or ''
                family = (
                    last_item.get('family_name') or last_item.get('familyName') or ''
                )
                if given or family:
                    author_full_name = f'{given} {family}'.strip()

        if author_full_name:
            elabftw_experiment.post_process(full_name=author_full_name)
        else:
            logger.warning('Could not extract author name from available keys')
    except Exception as e:
        logger.error(f'Could not extract the author name: {e}')

    mainfile_raw_path = os.path.dirname(mainfile)
    parent_folder_raw_path = mainfile.split('/')[-2]

    extracted_title = _set_child_entry_name(exp_id, exp_archive, archive, logger)

    matched = title_pattern.findall(extracted_title)
    if matched:
        title = matched[0]
    else:
        title = extracted_title
    elabftw_experiment.title = title

    try:
        export_json_path = [
            item['id']
            for item in raw_experiment['has_part']
            if item['id'].endswith('.json', None)
        ][0]
        path_to_export_json = os.path.join(mainfile_raw_path, export_json_path)

        with open(path_to_export_json) as f:
            export_data = json.load(f)
            export_data = (
                [export_data] if isinstance(export_data, dict) else export_data
            )
    except FileNotFoundError:
        raise ELabFTWParserError(
            f"Couldn't locate the {export_json_path} file of the main entry."
            f'Either the the exported file is corrupted or the data model is changed.'
        )

    def clean_nones(value):
        if isinstance(value, list):
            return [clean_nones(x) for x in value if x is not None]
        if isinstance(value, dict):
            return {k: clean_nones(v) for k, v in value.items() if v is not None}

        return value

    def convert_keys(target_dict: dict, conversion_mapping: dict) -> dict:
        new_dict = {}
        for key, value in target_dict.items():
            if isinstance(value, dict):
                value = convert_keys(value, conversion_mapping)
            elif isinstance(value, list):
                value = [
                    convert_keys(item, conversion_mapping)
                    if isinstance(item, dict)
                    else item
                    for item in value
                ]
            new_key = conversion_mapping.get(key, key)
            new_dict[new_key] = (
                {i: value[i] for i in range(len(value))}
                if isinstance(value, list) and new_key == 'extra_fields'
                else value
            )
        return new_dict

    key_mapping = {
        'extra': 'extra_fields',
        'link': 'experiments_links',
        'description': 'body',
        'name': 'title',
    }
    experiment_data = ELabFTWExperimentData()
    try:
        cleaned_data = clean_nones(export_data[0])
        cleaned_data = convert_keys(cleaned_data, key_mapping)

        # Normalize tags with backward compatibility
        # Try 'tags' first, then 'keywords' as fallback
        tags_value = cleaned_data.get('tags') or cleaned_data.get('keywords')
        if tags_value:
            if isinstance(tags_value, str):
                # Split pipe-separated or comma-separated strings
                if '|' in tags_value:
                    cleaned_data['tags'] = [
                        t.strip() for t in tags_value.split('|') if t.strip()
                    ]
                elif ',' in tags_value:
                    cleaned_data['tags'] = [
                        t.strip() for t in tags_value.split(',') if t.strip()
                    ]
                else:
                    # Single tag
                    cleaned_data['tags'] = (
                        [tags_value.strip()] if tags_value.strip() else []
                    )
            elif isinstance(tags_value, list):
                # Already a list, ensure it's clean
                cleaned_data['tags'] = [str(t).strip() for t in tags_value if t]
            # Remove 'keywords' if we used it as fallback
            if (
                'tags' in cleaned_data
                and 'keywords' in cleaned_data
                and cleaned_data.get('keywords') == tags_value
            ):
                cleaned_data.pop('keywords', None)

        experiment_data.m_update_from_dict(cleaned_data)
    except Exception as e:
        logger.error(f'Failed to parse experiment data: {e}')
        logger.warning(
            f"Couldn't read and parse the data from {export_json_path.lstrip('./')} file."
        )
    try:
        experiment_data.extra_fields = export_data[0]['metadata']['extra_fields']
    except Exception:
        pass
    elabftw_experiment.experiment_data = experiment_data

    try:
        exp_archive.metadata.comment = elabftw_entity_type

        lab_ids.extend(
            ('experiment_link', experiment_link['itemid'])
            for experiment_link in cleaned_data['experiments_links']
        )
        lab_ids.extend(
            ('item_link', experiment_link['itemid'])
            for experiment_link in cleaned_data['items_links']
        )
    except Exception:
        pass
    for file_id in raw_experiment['has_part']:
        file_section = _create_file_section(
            graph[file_id['id']], graph, parent_folder_raw_path, logger
        )
        elabftw_experiment.experiment_files.append(file_section)

    return elabftw_experiment


def _set_child_entry_name(exp_id, child_archive, archive, logger):
    matched_title = exp_id.split('/')
    if len(matched_title) > 1:
        extracted_title = matched_title[1]
        extracted_title = re.sub(r'[-_]', ' ', extracted_title)
        archive.metadata.m_update_from_dict(dict(entry_name='ELabFTW Schema'))
        child_archive.metadata.m_update_from_dict(dict(entry_name=extracted_title))
    else:
        logger.warning("Couldn't extract the title from experiment id")
        extracted_title = None
    return extracted_title


def _set_experiment_metadata(raw_experiment, exp_archive, elab_instance, logger):
    try:
        exp_external_id = raw_experiment['url'].split('&id=')[1]
        if match := re.search(r'.*/([^/]+)\.php', raw_experiment['url']):
            elabftw_entity_type = match.group(1)
        exp_archive.metadata.external_id = str(exp_external_id)
        elab_instance.project_id = exp_external_id
    except Exception:
        logger.error('Could not set the the external_id from the experiment url')
        elabftw_entity_type = None
    return elabftw_entity_type


def _parse_latest(
    graph,
    raw_experiment,
    exp_archive,
    mainfile,
    exp_id,
    archive,
    logger,
) -> ELabFTW:
    # Extract author with backward compatibility
    author_full_name = None
    try:
        # Strategy 1: Use the author reference from raw_experiment
        author_ref = raw_experiment.get('author', {})
        if isinstance(author_ref, dict):
            author_id = author_ref.get('@id') or author_ref.get('id')
        else:
            author_id = author_ref

        # Look up the author by ID in the graph
        if author_id and author_id in graph:
            author_obj = graph[author_id]

            # Try multiple key variants for given name
            given = (
                author_obj.get('givenName')
                or author_obj.get('given_name')
                or author_obj.get('firstName')
                or author_obj.get('name', '')
            )

            # Try multiple key variants for family name
            family = (
                author_obj.get('familyName')
                or author_obj.get('family_name')
                or author_obj.get('lastName')
                or author_obj.get('fullName', '')
            )

            author_full_name = f'{given} {family}'.strip()

        # Strategy 2: Fallback - search for Person or Author type in the graph
        if not author_full_name:
            author_types = ['Person', 'Author', 'schema:Person', 'schema:Author']
            for item in graph.values():
                if not isinstance(item, dict):
                    continue

                item_type = item.get('@type', item.get('type', ''))
                if isinstance(item_type, list):
                    matches_type = any(t in author_types for t in item_type)
                else:
                    matches_type = item_type in author_types

                if matches_type:
                    # Try multiple key variants for given name
                    given = (
                        item.get('givenName')
                        or item.get('given_name')
                        or item.get('firstName')
                        or item.get('name', '')
                    )

                    # Try multiple key variants for family name
                    family = (
                        item.get('familyName')
                        or item.get('family_name')
                        or item.get('lastName')
                        or item.get('fullName', '')
                    )

                    # If we found a person with name components, use them
                    if given or family:
                        author_full_name = f'{given} {family}'.strip()
                        break
    except Exception:
        logger.error('Could not extract the author name')

    # Process keywords - handle comma-separated string
    keywords_raw = raw_experiment.get('keywords', '')
    if isinstance(keywords_raw, str):
        keywords = [k.strip() for k in keywords_raw.split(',') if k.strip()]
    elif isinstance(keywords_raw, list):
        keywords = keywords_raw
    else:
        keywords = []

    latest_elab_instance = ELabFTW(
        author=author_full_name
        or (
            raw_experiment.get('author', {}).get('id')
            if isinstance(raw_experiment.get('author'), dict)
            else ''
        ),
        title=raw_experiment.get('name', None),
        keywords=keywords,
        id=raw_experiment.get('id', ''),
        status=raw_experiment.get('creative_work_status', 'Not set')
        if raw_experiment.get('creative_work_status', 'Not set') in status_types
        else 'Not set',
    )

    _ = _set_experiment_metadata(
        raw_experiment, exp_archive, latest_elab_instance, logger
    )
    data_section = ELabFTWExperimentData(
        body=raw_experiment.get('text', None),
        created_at=raw_experiment.get('date_created', None),
        extra_fields={
            i: value
            for i, value in enumerate(raw_experiment.get('variable_measured') or [])
        }
        if raw_experiment.get('variable_measured') is not None
        else {},
    )
    data_section.steps.extend(
        [
            ELabFTWSteps(
                ordering=step.get('position', None),
                deadline=step.get('expires', None),
                finished=step.get('creative_work_status', None) == 'finished',
                body=step.get('item_list_element', [{'text': None}])[0]['text'],
            )
            for step in raw_experiment.get('step', [])
        ]
    )

    data_section.experiments_links.extend(
        [
            ELabFTWExperimentLink(title=exp_link.get('id', None))
            for exp_link in raw_experiment.get('mentions', [])
        ]
    )

    parent_folder_raw_path = mainfile.split('/')[-2]
    for file_id in raw_experiment.get('has_part', []):
        file_section = _create_file_section(
            graph[file_id['id']], graph, parent_folder_raw_path, logger
        )
        latest_elab_instance.experiment_files.append(file_section)

    latest_elab_instance.m_add_sub_section(
        latest_elab_instance.m_def.all_sub_sections['experiment_data'], data_section
    )

    _ = _set_child_entry_name(exp_id, exp_archive, archive, logger)

    return latest_elab_instance
