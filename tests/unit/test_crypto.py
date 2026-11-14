import pytest
from cryptography.fernet import Fernet

from app.utils import crypto


def test_roundtrip():
    token = crypto.encrypt("sk-secret")
    assert token and token != "sk-secret"
    assert crypto.decrypt(token) == "sk-secret"


def test_empty_values_pass_through():
    assert crypto.encrypt(None) is None
    assert crypto.encrypt("") is None
    assert crypto.decrypt(None) is None


def test_wrong_key_raises_clear_error(set_env):
    token = crypto.encrypt("sk-secret")
    set_env(SECRET_KEY=Fernet.generate_key().decode())
    with pytest.raises(crypto.DecryptionError, match="SECRET_KEY"):
        crypto.decrypt(token)


def test_prod_without_key_refuses(set_env):
    set_env(APP_ENV="prod", SECRET_KEY="")
    with pytest.raises(crypto.SecretKeyMissingError):
        crypto.encrypt("x")


@pytest.mark.parametrize(
    ("value", "masked"),
    [("sk-proj-abcdefgh1234", "sk-…1234"), ("short", "…rt"), (None, None)],
)
def test_mask(value, masked):
    assert crypto.mask_secret(value) == masked
