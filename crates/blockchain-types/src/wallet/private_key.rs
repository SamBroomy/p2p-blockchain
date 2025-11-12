use ed25519_dalek::{Signature, SigningKey, ed25519::signature::SignerMut};
use secrecy::{ExposeSecret, SecretSlice};

#[derive(Debug, Clone)]
pub struct PrivateKey(SecretSlice<u8>);

impl PrivateKey {
    pub fn from_bytes(bytes: [u8; 32]) -> Self {
        let secret_box = SecretSlice::new(Box::new(bytes));
        Self(secret_box)
    }

    #[allow(clippy::needless_pass_by_value)]
    pub fn from_singing_key(sk: SigningKey) -> Self {
        Self::from_bytes(sk.to_bytes())
    }

    fn to_signing_key(&self) -> SigningKey {
        let bytes: &[u8; 32] = self
            .0
            .expose_secret()
            .try_into()
            .expect("Should always be a valid 32-byte array");
        SigningKey::from_bytes(bytes)
    }

    pub fn sign(&self, msg: &[u8]) -> Signature {
        let mut signing_key = self.to_signing_key();
        signing_key.sign(msg)
    }
}

impl PartialEq for PrivateKey {
    fn eq(&self, other: &Self) -> bool {
        self.0.expose_secret() == other.0.expose_secret()
    }
}

impl Eq for PrivateKey {}

impl AsRef<[u8]> for PrivateKey {
    fn as_ref(&self) -> &[u8] {
        self.0.expose_secret()
    }
}
impl AsRef<SecretSlice<u8>> for PrivateKey {
    fn as_ref(&self) -> &SecretSlice<u8> {
        &self.0
    }
}
impl From<SigningKey> for PrivateKey {
    fn from(sk: SigningKey) -> Self {
        Self::from_singing_key(sk)
    }
}
impl From<PrivateKey> for SigningKey {
    fn from(pk: PrivateKey) -> Self {
        pk.to_signing_key()
    }
}
impl Default for PrivateKey {
    fn default() -> Self {
        let mut csprng = rand::rngs::OsRng;
        let sk = SigningKey::generate(&mut csprng);
        Self::from_singing_key(sk)
    }
}
