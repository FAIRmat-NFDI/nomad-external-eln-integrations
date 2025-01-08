from nomad.config.models.plugins import ParserEntryPoint


class ElabftwEntryPoint(ParserEntryPoint):
    def load(self):
        from nomad_external_eln_integrations.parsers.elabftw import ELabFTWParser

        return ELabFTWParser(**self.dict())


elabftw_parser_entry_point = ElabftwEntryPoint(
    name='parsers/elabftw',
    aliases=['parsers/elabftw', 'parsers/elabftw_parser'],
    code_name='elabftw',
    code_homepage='https://www.elabftw.net/',
    description='NOMAD parser for eln file formats.',
    mainfile_mime_re=r'text/plain|application/json|text/html',
    mainfile_name_re=r'.*ro-crate-metadata.json$',
)


class ChemotionEntryPoint(ParserEntryPoint):
    def load(self):
        from nomad_external_eln_integrations.parsers.chemotion import ChemotionParser

        return ChemotionParser(**self.dict())


chemotion_parser_entry_point = ElabftwEntryPoint(
    name='parsers/chemotion',
    aliases=['parsers/chemotion'],
    code_name='chemotion',
    code_homepage='https://chemotion.net/',
    description='NOMAD parser for chemotion data.',
    mainfile_mime_re=r'application/json|text/plain',
    mainfile_name_re=r'^.*export.json$',
)
