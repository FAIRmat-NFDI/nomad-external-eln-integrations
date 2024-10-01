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

import numpy as np
import pytest
from nomad.datamodel import EntryArchive, EntryMetadata

from src.nomad_external_eln_integrations.parsers.chemotion.parser import (
    ChemotionParser,
    _element_type_section_mapping,
)


@pytest.fixture(scope='module')
def parser():
    return ChemotionParser()


def _assert_chemotion(test_archive):
    assert test_archive.data is not None
    assert test_archive.data.Collection is not None

    assert test_archive.data.Collection[0].label == 'Modification Sequence'
    assert (
        test_archive.data.Collection[0].user_id
        == '60c41de1-b83d-4487-a599-8eb310847b8a'
    )
    assert test_archive.data.Collection[0].is_locked is False

    assert len(test_archive.data.Sample) == 4
    assert test_archive.data.Sample[0].xref == {
        'cas': {'label': '554-95-0', 'value': '554-95-0'}
    }
    assert test_archive.data.Sample[1].name == 'Aqua dest.'
    assert test_archive.data.Sample[2].target_amount_value == np.float16(0.001)
    assert test_archive.data.Sample[3].target_amount_value == np.float16(0.002)

    assert len(test_archive.data.CollectionsSample) == 4
    assert len(test_archive.data.Fingerprint) == 3
    assert len(test_archive.data.Molecule) == 3
    assert test_archive.data.Molecule[1].inchikey == 'XLYOFNOQVPJJNP-UHFFFAOYSA-N'

    for k in _element_type_section_mapping.keys():
        k = 'Reactions' if k == 'Reaction' else k
        assert k in test_archive.data.m_def.all_properties
        assert test_archive.data.m_def.all_properties[k] is not None


def test_chemotion(parser):
    mainfile = 'tests/data/parsers/chemotion/test/export.json'
    archive = EntryArchive(metadata=EntryMetadata())
    child_archive = {
        f'{i}': EntryArchive(metadata=EntryMetadata()) for i in range(0, 1)
    }
    parser.parse(
        mainfile,
        archive,
        None,
        child_archive,
    )
    child_archive = child_archive['0']
    _assert_chemotion(child_archive)
