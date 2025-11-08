use ed25519_dalek::{Signature, SigningKey, ed25519::signature::SignerMut};
use secrecy::{ExposeSecret, SecretSlice};

#[derive(Debug, Clone)]
pub(crate) struct PrivateKey(SecretSlice<u8>);

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
            .expect("Invalid key length");
        SigningKey::from_bytes(bytes)
    }

    pub fn sign(&self, msg: &[u8]) -> Signature {
        let mut signing_key = self.to_signing_key();
        signing_key.sign(msg)
    }
}
