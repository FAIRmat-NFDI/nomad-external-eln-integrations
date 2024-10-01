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

import pytest
from nomad.datamodel import EntryArchive, EntryMetadata

from src.nomad_external_eln_integrations.parsers.elabftw import ELabFTWParser


@pytest.fixture(scope='module')
def parser():
    return ELabFTWParser()


@pytest.mark.parametrize(
    'mainfile, no_child_archives, expected_results',
    [
        pytest.param(
            'tests/data/parsers/elabftw/legacy/ro-crate-metadata.json',
            1,
            {
                'expected_title': 'Test',
                'expected_id': 'ro-crate-metadata.json',
                'expected_experiments_links': 1,
                'expected_link_title': 'Untitled',
                'expected_experiment_title': 'JSON test ',
                'expected_files': 5,
            },
            id='legacy_data_model',
        ),
        pytest.param(
            'tests/data/parsers/elabftw/with_file/2024-09-19-151520-export/ro-crate-metadata.json',
            3,
            {
                'expected_title': 'new experiment',
                'expected_id': './new-experiment - 582d690f/',
                'expected_experiments_links': 2,
                'expected_link_title': './Tests - Accusamus-dolor-numquam-ducimus-dolorum-sunt - 8d813331/',
                'expected_experiment_title': './Tests - Accusamus-dolor-numquam-ducimus-dolorum-sunt - 8d813331/',
                'expected_files': 1,
            },
            id='latest_data_model',
        ),
    ],
)
def test_elabftw(parser, mainfile, no_child_archives: int, expected_results):
    archive = EntryArchive(metadata=EntryMetadata())
    child_archive = {
        f'{i}': EntryArchive(metadata=EntryMetadata())
        for i in range(0, no_child_archives)
    }
    parser.parse(
        mainfile,
        archive,
        None,
        child_archive,
    )
    child_archive = child_archive['0']
    assert child_archive.data is not None
    assert child_archive.data.title == expected_results['expected_title']
    assert child_archive.data.id == expected_results['expected_id']
    assert (
        len(child_archive.data.experiment_data.experiments_links)
        == expected_results['expected_experiments_links']
    )

    assert (
        len(child_archive.data.experiment_files) == expected_results['expected_files']
    )

    assert (
        child_archive.data.experiment_data.experiments_links[0].title
        == expected_results['expected_experiment_title']
    )
    assert (
        (
            child_archive.data.experiment_data.items_links[0].title
            == expected_results['expected_link_title']
        )
        if len(child_archive.data.experiment_data.items_links) != 0
        else True
    )

    assert child_archive.data.experiment_data.extra_fields is not None
    if not expected_results['expected_files']:
        for item in child_archive.data.experiment_files:
            assert item.type == 'File'
            assert item.file is not None
            assert item.id is not None
