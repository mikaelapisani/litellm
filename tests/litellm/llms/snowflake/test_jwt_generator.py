import pytest
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.primitives import serialization

from litellm.llms.snowflake.jwt_generator import JWTGenerator


@pytest.fixture
def rsa_private_key_pem():
    """Generates a temporary RSA private key in PEM format."""
    private_key = rsa.generate_private_key(
        public_exponent=65537,
        key_size=2048,
    )
    pem = private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )
    return pem.decode("utf-8")


def test_prepare_account_name_for_jwt_non_global():
    result = JWTGenerator.prepare_account_name_for_jwt("myaccount.domain.com")
    assert result == "MYACCOUNT"
    result = JWTGenerator.prepare_account_name_for_jwt("myorg-myaccount.us-west-2.aws")
    assert result == "MYORG-MYACCOUNT"


def test_prepare_account_name_for_jwt_global():
    result = JWTGenerator.prepare_account_name_for_jwt("globalaccount-global")
    assert result == "GLOBALACCOUNT"


def test_jwt_generator_initialization(rsa_private_key_pem):
    jwt_gen = JWTGenerator(
        account="myaccount.domain.com",
        user="testuser",
        private_key_string=rsa_private_key_pem,
        passphrase="",
    )

    assert jwt_gen.account == "MYACCOUNT"
    assert jwt_gen.user == "TESTUSER"
    assert jwt_gen.qualified_username == "MYACCOUNT.TESTUSER"
    assert jwt_gen.private_key is not None


@patch("litellm.llms.snowflake.jwt_generator.datetime")
def test_get_token_creates_token(mock_datetime, rsa_private_key_pem):
    now = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    mock_datetime.now.return_value = now
    mock_datetime.side_effect = lambda *args, **kwargs: datetime(*args, **kwargs)

    jwt_gen = JWTGenerator(
        account="myaccount.domain.com",
        user="testuser",
        private_key_string=rsa_private_key_pem,
        passphrase="",
    )

    token = jwt_gen.get_token()

    assert token is not None

    # A JWT has 3 parts separated by dots.
    assert len(token.split(".")) == 3


@patch("litellm.llms.snowflake.jwt_generator.datetime")
def test_get_token_reuses_existing_token(mock_datetime, rsa_private_key_pem):
    now = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    mock_datetime.now.return_value = now
    mock_datetime.side_effect = lambda *args, **kwargs: datetime(*args, **kwargs)

    jwt_gen = JWTGenerator(
        account="myaccount.domain.com",
        user="testuser",
        private_key_string=rsa_private_key_pem,
        passphrase="",
    )

    first_token = jwt_gen.get_token()

    # Move time forward but not past renewal_delay
    mock_datetime.now.return_value = now + timedelta(minutes=10)
    second_token = jwt_gen.get_token()

    # Token should be reused because we didn't hit the renewal time
    assert first_token == second_token


@patch("litellm.llms.snowflake.jwt_generator.datetime")
def test_get_token_renews_after_renewal_delay(mock_datetime, rsa_private_key_pem):
    now = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    mock_datetime.now.return_value = now
    mock_datetime.side_effect = lambda *args, **kwargs: datetime(*args, **kwargs)

    jwt_gen = JWTGenerator(
        account="myaccount.domain.com",
        user="testuser",
        private_key_string=rsa_private_key_pem,
        passphrase="",
    )

    first_token = jwt_gen.get_token()

    # Move time past renewal_delay
    mock_datetime.now.return_value = now + jwt_gen.renewal_delay + timedelta(minutes=1)
    second_token = jwt_gen.get_token()

    # Token should be renewed
    assert first_token != second_token


def test_calculate_public_key_fingerprint(rsa_private_key_pem):
    jwt_gen = JWTGenerator(
        account="myaccount.domain.com",
        user="testuser",
        private_key_string=rsa_private_key_pem,
        passphrase="",
    )

    fingerprint = jwt_gen.calculate_public_key_fingerprint(jwt_gen.private_key)

    assert fingerprint.startswith("SHA256:")
    # The fingerprint part should be base64, ensure it's non-empty
    assert len(fingerprint) > len("SHA256:")
