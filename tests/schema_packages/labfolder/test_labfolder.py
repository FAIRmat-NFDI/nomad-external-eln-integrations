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
from unittest.mock import MagicMock

import pytest
import requests
from nomad import utils
from nomad.datamodel import EntryArchive, EntryMetadata

from nomad_external_eln_integrations.schema_packages.labfolder.schema import (
    LabfolderImportError,
    LabfolderProject,
)


def test_labfolder_integration():
    filename = 'tests/data/schemas/labfolder/example-labfolder.archive.json'
    data = json.load(open(filename))
    test_archive = EntryArchive(metadata=EntryMetadata())
    test_archive.data = LabfolderProject()
    data['data'].pop('m_def')
    test_archive.data.m_update_from_dict(data['data'])

    assert test_archive.data.project_url is not None
    assert len(test_archive.data.entries) == 6


@pytest.mark.parametrize(
    'status_code,project_url,labfolder_email,password,element_data,response_data',
    [
        pytest.param(
            200,
            'https://labfolder.labforward.app/eln/notebook#?projectIds=1',
            'test_email',
            'test_password',
            [{'elements': [{'id': '1', 'version_id': '1', 'type': 'TEXT'}]}],
            {
                'entry_id': '1',
                'id': '1',
                'version_id': '1',
                'creation_date': '2022-08-29T09:22:30+00:00',
                'version_date': '2022-08-29T09:22:30+00:00',
                'owner_id': '1',
                'element_type': 'TEXT',
                'content': '<p>test content</p>',
            },
            id='TEXT_element',
        ),
        pytest.param(
            200,
            'https://labfolder.labforward.app/eln/notebook#?projectIds=1',
            'test_email',
            'test_password',
            [{'elements': [{'id': '1', 'version_id': '1', 'type': 'TABLE'}]}],
            {
                'entry_id': '1',
                'id': '1',
                'version_id': '1',
                'creation_date': '2022-08-29T09:22:33+00:00',
                'version_date': '2022-08-29T09:22:33+00:00',
                'owner_id': '1',
                'element_type': 'TABLE',
                'title': 'Table Title',
                'content': {'version': '14.1.3', 'nested_content': 'more content'},
            },
            id='TABLE_element',
        ),
        pytest.param(
            200,
            'https://labfolder.labforward.app/eln/notebook#?projectIds=1',
            'test_email',
            'test_password',
            [{'elements': [{'id': '1', 'version_id': '1', 'type': 'WELL_PLATE'}]}],
            {
                'entry_id': '1',
                'id': '1',
                'version_id': '1',
                'creation_date': '2022-08-29T09:22:33+00:00',
                'version_date': '2022-08-29T09:22:33+00:00',
                'owner_id': '1',
                'element_type': 'WELL_PLATE',
                'title': 'Wellplate title',
                'content': {'version': '14.1.3', 'nested_content': 'more _content'},
                'meta_data': {
                    'plate': {'size': '24'},
                    'activeLayer': {'name': 'Composite', 'type': 'COMPOSITE'},
                },
            },
            id='WELL_PLATE_element',
        ),
        pytest.param(
            200,
            'https://labfolder.labforward.app/eln/notebook#?projectIds=1',
            'test_email',
            'test_password',
            [{'elements': [{'id': '1', 'version_id': '1', 'type': 'DATA'}]}],
            {
                'entry_id': '1',
                'id': '1',
                'version_id': '1',
                'creation_date': '2022-08-29T09:22:33+00:00',
                'version_date': '2022-08-29T09:22:34+00:00',
                'owner_id': '1',
                'element_type': 'DATA',
                'data_elements': [
                    {
                        'type': 'DATA_ELEMENT_GROUP',
                        'title': 'first title',
                        'children': [
                            {
                                'type': 'SINGLE_DATA_ELEMENT',
                                'title': 'child title',
                                'value': '1',
                                'unit': 'mL',
                                'physical_quantity_id': '6',
                            },
                            {
                                'type': 'DESCRIPTIVE_DATA_ELEMENT',
                                'title': 'Water',
                                'description': 'add up to 1 liter',
                            },
                        ],
                    }
                ],
            },
            id='DATA_element',
        ),
        pytest.param(
            400,
            'https://labfolder.labforward.app/eln/notebook#?projectIds=1',
            None,
            'test_password',
            None,
            None,
            id='missing_email',
        ),
        pytest.param(
            400,
            'https://labfolder.labforward.app/eln/notebook#?projectIds=1',
            'test_email',
            None,
            None,
            None,
            id='missing_password',
        ),
        pytest.param(
            400,
            None,
            'test_email',
            'test_password',
            None,
            None,
            id='missing_project_url',
        ),
        pytest.param(
            400, 'wrong_url', 'test_email', 'test_password', None, None, id='wrong_url'
        ),
        pytest.param(
            400,
            'https://labfolder.labforward.app/eln/notebook#?projectIds=1',
            'test_email',
            'test_password',
            None,
            None,
            id='wrong_element_data',
        ),
    ],
)
def test_labfolder_detailed(
    monkeypatch,
    status_code,
    project_url,
    labfolder_email,
    password,
    element_data,
    response_data,
):
    logger = utils.get_logger(__name__)
    base_api_url = 'https://labfolder.labforward.app/api/v2'

    def mock_labfolder_json_method(*args, **kwargs):
        mock_response = MagicMock()
        mock_response.status_code = status_code
        mock_response.url = '/test_endpoint'

        if args[0] == f'{base_api_url}/entries?project_ids=1':
            mock_response.return_value = element_data
        else:
            mock_response.return_value = response_data

        mock_response.json.return_value = mock_response.return_value
        return mock_response

    def mock_labfolder_response_method(*args, **kwargs):
        mock_response = MagicMock()
        mock_response.status_code = status_code
        mock_response.json.return_value = {'token': 'test'}
        return mock_response

    # patching API calls to labfolder server
    monkeypatch.setattr(requests, 'post', mock_labfolder_response_method)
    monkeypatch.setattr(requests, 'get', mock_labfolder_json_method)

    # Creating labfolder instance
    test_archive = EntryArchive(metadata=EntryMetadata())
    labfolder_instance = LabfolderProject()
    labfolder_instance.project_url = project_url
    labfolder_instance.labfolder_email = labfolder_email
    labfolder_instance.password = password
    labfolder_instance.resync_labfolder_repository = True
    test_archive.data = labfolder_instance

    if status_code == 200:
        labfolder_instance.normalize(test_archive, logger=logger)
        assert len(labfolder_instance.entries) == 1
        assert labfolder_instance.labfolder_email is None
        assert labfolder_instance.password is None

        parsed_data = labfolder_instance.entries[0].elements[0].m_to_dict()
        del parsed_data['m_def']
        if parsed_data['element_type'] != 'DATA':
            assert json.dumps(parsed_data, sort_keys=True) == json.dumps(
                response_data, sort_keys=True
            )
        else:
            del parsed_data['labfolder_data']
            assert json.dumps(parsed_data, sort_keys=True) == json.dumps(
                response_data, sort_keys=True
            )
    else:
        with pytest.raises(LabfolderImportError):
            labfolder_instance.normalize(test_archive, logger=logger)


def test_labfolder_unauthorized(monkeypatch):
    logger = utils.get_logger(__name__)
    project_url = 'https://labfolder.labforward.app/eln/notebook#?projectIds=1'
    labfolder_email = 'test_email'
    password = 'test_password'

    def mock_labfolder_response_method(*args, **kwargs):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {'token': 'test'}
        return mock_response

    mock_response_401 = MagicMock()
    mock_response_401.status_code = 401

    mock_response_200 = MagicMock()
    mock_response_200.status_code = 200
    mock_response_200.json.return_value = [
        {'elements': [{'id': '1', 'version_id': '1', 'type': 'TEXT'}]}
    ]

    mock_response_element = MagicMock()
    mock_response_element.status_code = 200
    mock_response_element.json.return_value = {
        'entry_id': '1',
        'id': '1',
        'version_id': '1',
        'element_type': 'TEXT',
        'content': '<p>test content</p>',
    }

    def mock_labfolder_get_method(url, headers=None, **kwargs):
        if 'entries' in url:
            if headers and 'Bearer' in headers.get('Authorization', ''):
                return mock_response_401
            return mock_response_200
        return mock_response_element

    monkeypatch.setattr(requests, 'post', mock_labfolder_response_method)
    monkeypatch.setattr(requests, 'get', mock_labfolder_get_method)

    test_archive = EntryArchive(metadata=EntryMetadata())
    labfolder_instance = LabfolderProject()
    labfolder_instance.project_url = project_url
    labfolder_instance.labfolder_email = labfolder_email
    labfolder_instance.password = password
    labfolder_instance.resync_labfolder_repository = True
    test_archive.data = labfolder_instance

    labfolder_instance.normalize(test_archive, logger=logger)
    assert len(labfolder_instance.entries) == 1
