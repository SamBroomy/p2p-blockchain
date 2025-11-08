use ed25519_dalek::{Signature, SigningKey, ed25519::signature::SignerMut};
use rand::rngs::OsRng;
use secrecy::{ExposeSecret, SecretSlice};

use crate::blockchain::{Address, Transaction, transaction::TransactionConstructor};

#[derive(Debug, Clone)]
pub(super) struct PrivateKey(SecretSlice<u8>);

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

#[derive(Debug, Clone)]
pub struct Wallet {
    private_key: PrivateKey,
    public_key: Address,
}

impl Wallet {
    pub fn from_singing_key(sk: SigningKey) -> Self {
        let public_key = Address::from_verifying_key(sk.verifying_key());
        let private_key = PrivateKey::from_singing_key(sk);
        Self {
            private_key,
            public_key,
        }
    }

    pub fn new() -> Self {
        let mut csprng = OsRng;
        Self::from_singing_key(SigningKey::generate(&mut csprng))
    }

    /// Deterministic key generation from password/seed
    pub fn from_seed(seed: &str) -> Self {
        debug_assert!(!seed.is_empty(), "Seed must not be empty");
        // Hash the seed to get 32 bytes
        let hash = blake3::hash(seed.as_bytes());
        let bytes: [u8; 32] = hash.into();
        // Generate key from those 32 bytes
        Self::from_singing_key(SigningKey::from_bytes(&bytes))
    }

    pub fn address(&self) -> &Address {
        &self.public_key
    }

    pub fn create_transaction(&self, receiver_address: &Address, amount: u64) -> Transaction {
        debug_assert!(amount > 0, "Transaction amount must be > 0");
        debug_assert_ne!(
            self.public_key.as_bytes(),
            receiver_address.as_bytes(),
            "Cannot send transaction to self"
        );

        TransactionConstructor::new_transaction(
            &self.public_key,
            receiver_address,
            amount,
            &self.private_key,
        )
    }
}

impl Default for Wallet {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_wallet_creation() {
        let wallet = Wallet::new();
        let address = wallet.address();

        assert!(!address.as_bytes().is_empty());
    }

    #[test]
    fn test_wallet_from_seed() {
        let seed = "my_secure_seed";
        let wallet1 = Wallet::from_seed(seed);
        let wallet2 = Wallet::from_seed(seed);

        assert_eq!(wallet1.address(), wallet2.address());
    }

    #[test]
    fn test_wallet_unique_addresses() {
        let wallet1 = Wallet::new();
        let wallet2 = Wallet::new();

        assert_ne!(wallet1.address(), wallet2.address());
    }

    #[test]
    fn test_wallet_create_transaction() {
        let sender = Wallet::new();
        let receiver = Wallet::new();

        let tx = sender.create_transaction(receiver.address(), 100);

        assert_eq!(tx.sender(), sender.address());
        assert_eq!(tx.receiver(), receiver.address());
        assert_eq!(tx.amount(), 100);
        assert!(tx.validate().is_ok());
    }
}
