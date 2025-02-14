# Author: Sean Iannuzzi
# Updated: January 14, 2024

import base64
import hashlib
from datetime import datetime, timedelta, timezone
from typing import Optional

import jwt
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives.serialization import (
    Encoding,
    PublicFormat,
    load_pem_private_key,
)

# Constants for JWT claims
ISSUER = "iss"  # Issuer of the JWT
EXPIRE_TIME = "exp"  # Expiration time for the JWT
ISSUE_TIME = "iat"  # Issue time for the JWT
SUBJECT = "sub"  # Subject of the JWT (usually the user or service)


class JWTGenerator(object):
    """
    JWTGenerator class to generate JSON Web Tokens using RSA private keys.
    The class allows generating tokens with a specified lifetime and renewal delay.
    
    Attributes:
        account (str): The account name (e.g., organization or application name).
        user (str): The user identifier (e.g., service account or user ID).
        private_key_string (str): The private key as a string for JWT signing.
        passphrase (str):  Passphrase used to decrypt the private key, if it's encrypted.
        lifetime (Optional[timedelta]): Token's lifetime duration (default is 59 minutes).
        renewal_delay (Optionaltimedelta]): Time before the token needs to be renewed (default is 54 minutes).
    """

    def __init__(
        self,
        account: str,
        user: str,
        private_key_string: str,
        passphrase: str,
        algorithm: str = "RS256",
        lifetime: Optional[timedelta] = timedelta(minutes=59),
        renewal_delay: Optional[timedelta] = timedelta(minutes=54),
    ):
        self.account = self.prepare_account_name_for_jwt(account)
        self.user = user.upper()  # Ensure the username is in uppercase
        self.qualified_username = (
            self.account + "." + self.user
        )  # Full qualified username
        self.lifetime = lifetime  # Token's lifetime
        self.renewal_delay = renewal_delay  # Renewal time for token
        self.private_key_string = private_key_string  # The private key string
        self.passphrase = (
            passphrase  # Passphrase for private key decryption (if needed)
        )
        self.algorithm = algorithm
        self.renew_time = datetime.now(
            timezone.utc
        )  # Time when the token needs to be renewed
        self.token = None  # Placeholder for the generated JWT

        # Load the private key from the provided string and passphrase
        password_to_use = (
            self.passphrase.encode() if self.passphrase else None
        )  # Ensure passphrase is encoded if provided
        self.private_key = load_pem_private_key(
            self.private_key_string.encode(), password_to_use, default_backend()
        )  # Load and decrypt the private key

    def prepare_account_name_for_jwt(self, raw_account):
        """
        Prepares the account name by stripping off additional parts to standardize it for JWT.

        :param raw_account: The raw account name.
        :return: The cleaned account name.
        """
        account = raw_account
        if ".global" not in account:
            idx = account.find(".")
            if idx > 0:
                account = account[
                    :idx
                ]  # Extract the part before the first dot if it is not global
        else:
            idx = account.find("-")
            if idx > 0:
                account = account[
                    :idx
                ]  # Extract the part before the first hyphen if it's a global account
        return account.upper()  # Return the standardized account name in uppercase

    def get_token(self):
        """
        Generates a JWT token using the private key if it's not expired or needs renewal.

        :return: The generated JWT as a string.
        """
        now = datetime.now(timezone.utc)  # Current UTC time
        if not self.lifetime or not self.renewal_delay:
            return self.token
        # Renew the token if it's not set or if it's time to renew
        if self.token is None or self.renew_time <= now:
            self.renew_time = now + self.renewal_delay  # Update the renewal time
            public_key_fp = self.calculate_public_key_fingerprint(
                self.private_key
            )  # Generate the public key fingerprint

            # Prepare the JWT payload with necessary claims
            payload = {
                ISSUER: self.qualified_username
                + "."
                + public_key_fp,  # Issuer with public key fingerprint
                SUBJECT: self.qualified_username,  # Subject (user or service account)
                ISSUE_TIME: now,  # Issue time (current time)
                EXPIRE_TIME: now
                + self.lifetime,  # Expiration time (current time + lifetime)
            }

            # Encode the payload using the private key and the RS256 algorithm
            token = jwt.encode(payload, key=self.private_key_string, algorithm=self.algorithm)
            if isinstance(token, bytes):
                token = token.decode(
                    "utf-8"
                )  # Decode the token to string if it's bytes
            self.token = token  # Store the generated token
        return self.token  # Return the generated JWT

    def calculate_public_key_fingerprint(self, private_key):
        """
        Calculates the SHA-256 fingerprint of the public key extracted from the private key.

        :param private_key: The private key object.
        :return: The fingerprint in SHA256 base64 encoded format.
        """
        public_key_raw = private_key.public_key().public_bytes(
            Encoding.DER, PublicFormat.SubjectPublicKeyInfo
        )  # Extract the public key in DER format
        sha256hash = hashlib.sha256()  # Create a SHA-256 hash object
        sha256hash.update(public_key_raw)  # Update the hash with the public key data
        public_key_fp = "SHA256:" + base64.b64encode(sha256hash.digest()).decode(
            "utf-8"
        )  # Generate the fingerprint and encode in base64
        return public_key_fp  # Return the fingerprint
