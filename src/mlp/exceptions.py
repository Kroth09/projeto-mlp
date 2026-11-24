"""Exceções específicas do domínio da aplicação."""


class MLPError(Exception):
    """Erro base da aplicação."""


class ConfigurationError(MLPError):
    """Configuração ausente, inválida ou incompatível."""


class DatasetError(MLPError):
    """Dataset inválido ou incompatível com a rede."""


class NormalizationError(MLPError):
    """Normalização ausente, inválida ou incompatível."""


class PersistenceError(MLPError):
    """Falha ao salvar ou reconstruir uma rede."""

