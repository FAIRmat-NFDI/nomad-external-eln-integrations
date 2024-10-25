from nomad.config.models.plugins import SchemaPackageEntryPoint


class OpenbisEntryPoint(SchemaPackageEntryPoint):
    def load(self):
        from nomad_external_eln_integrations.schema_packages.openbis.schema import (
            m_package,
        )

        return m_package


openbis_schema = OpenbisEntryPoint(
    name='openbis',
    description='NOMAD integration for mapping Openbis data to NOMAD schema',
)
