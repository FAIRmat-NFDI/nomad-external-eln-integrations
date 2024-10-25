from nomad.config.models.plugins import SchemaPackageEntryPoint


class LabfolderEntryPoint(SchemaPackageEntryPoint):
    def load(self):
        from nomad_external_eln_integrations.schema_packages.labfolder.schema import (
            m_package,
        )

        return m_package


labfolder_schema = LabfolderEntryPoint(
    name='labfolder',
    description='NOMAD integration for mapping Labfolder data to NOMAD schema',
)
