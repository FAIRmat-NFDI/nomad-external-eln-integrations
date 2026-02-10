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
    'mainfile, no_child_archives, test_child_index, expected_results',
    [
        pytest.param(
            'tests/data/parsers/elabftw/legacy/ro-crate-metadata.json',
            1,
            0,  # Only one experiment
            {
                'expected_title': '2023 01 13   Test   86506194',
                'expected_id': 'ro-crate-metadata.json',
                'expected_author': 'Demo User',  # Tests givenName/familyName extraction
                'expected_tags': None,  # Tags are null in legacy data
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
            1,  # Test the second experiment which has keywords (comma-separated)
            {
                'expected_title': 'Accusamus dolor numquam ducimus dolorum sunt.',
                'expected_id': './Tests - Accusamus-dolor-numquam-ducimus-dolorum-sunt - 8d813331/',
                'expected_author': 'Nicolas CARPi',  # Tests givenName/familyName in latest format
                'expected_keywords': [
                    'Software',
                    'lab supplies',
                ],  # Tests comma-separated keywords
                'expected_experiments_links': 0,
                'expected_link_title': '',
                'expected_experiment_title': '',
                'expected_files': 0,
            },
            id='latest_data_model',
        ),
        pytest.param(
            'tests/data/parsers/elabftw/backward_compat/ro-crate-metadata.json',
            1,
            0,  # Only one experiment
            {
                'expected_title': '2024 01 01   Test Experiment   abc123',
                'expected_id': 'ro-crate-metadata.json',
                'expected_author': 'Test User',  # Tests givenName/familyName fallback
                'expected_tags': [
                    'catalysis',
                    'test-sample',
                    'oxidation',
                ],  # Tests pipe-separated tags
                'expected_experiments_links': 0,
                'expected_link_title': 'Test Material',
                'expected_experiment_title': '',
                'expected_files': 1,
            },
            id='backward_compat',
        ),
    ],
)
def test_elabftw(
    parser, mainfile, no_child_archives: int, test_child_index: int, expected_results
):
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
    child_archive = child_archive[str(test_child_index)]
    assert child_archive.data is not None
    assert child_archive.data.title == expected_results['expected_title']
    assert child_archive.data.id == expected_results['expected_id']

    # Test author extraction across all formats
    if 'expected_author' in expected_results:
        assert (
            child_archive.data.author == expected_results['expected_author']
        ), f"Author mismatch: expected '{expected_results['expected_author']}', got '{child_archive.data.author}'"

    # Test tags normalization (pipe-separated in backward_compat, null in legacy)
    if 'expected_tags' in expected_results:
        actual_tags = child_archive.data.experiment_data.tags
        expected_tags = expected_results['expected_tags']
        assert (
            actual_tags == expected_tags
        ), f'Tags mismatch: expected {expected_tags}, got {actual_tags}'

    # Test keywords extraction (comma-separated in latest format)
    if 'expected_keywords' in expected_results:
        actual_keywords = child_archive.data.keywords
        expected_keywords = expected_results['expected_keywords']
        assert (
            actual_keywords == expected_keywords
        ), f'Keywords mismatch: expected {expected_keywords}, got {actual_keywords}'

    assert (
        len(child_archive.data.experiment_data.experiments_links)
        == expected_results['expected_experiments_links']
    )

    assert (
        len(child_archive.data.experiment_files) == expected_results['expected_files']
    )

    # Skip if no experiments_links expected
    if expected_results['expected_experiments_links'] > 0:
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
