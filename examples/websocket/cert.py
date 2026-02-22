import datetime
import ipaddress
import sys
from cryptography import x509
from cryptography.x509.oid import NameOID
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa

if len(sys.argv) < 2:
    print(f"Usage: python {sys.argv[0]} <host_or_ip>")
    print(f"Example: python {sys.argv[0]} 192.168.1.123")
    sys.exit(1)

host = sys.argv[1]

try:
    addr = ipaddress.ip_address(host)
    san = [x509.IPAddress(addr)]
except ValueError:
    san = [x509.DNSName(host)]

key = rsa.generate_private_key(public_exponent=65537, key_size=2048)

cert = (
    x509.CertificateBuilder()
    .subject_name(x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, host)]))
    .issuer_name(x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, host)]))
    .public_key(key.public_key())
    .serial_number(x509.random_serial_number())
    .not_valid_before(datetime.datetime.now(datetime.timezone.utc))
    .not_valid_after(datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(days=365))
    .add_extension(
        x509.SubjectAlternativeName(san),
        critical=False,
    )
    .sign(key, hashes.SHA256())
)

with open(f"{host}.pem", "wb") as f:
    f.write(cert.public_bytes(serialization.Encoding.PEM))

with open(f"{host}-key.pem", "wb") as f:
    f.write(key.private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.TraditionalOpenSSL,
        serialization.NoEncryption(),
    ))

print(f"Generated: {host}.pem, {host}-key.pem")
