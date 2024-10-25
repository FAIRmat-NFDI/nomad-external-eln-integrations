from nomad.config.models.plugins import SchemaPackageEntryPoint


class ElabftwNormalizerEntryPoint(SchemaPackageEntryPoint):
    def load(self):
        from nomad_external_eln_integrations.schema_packages.elabftw.schema import (
            m_package,
        )

        return m_package


elabftw_schema = ElabftwNormalizerEntryPoint(
    name='elabftw',
    description='NOMAD integration for mapping elabftw data to NOMAD schema',
)
