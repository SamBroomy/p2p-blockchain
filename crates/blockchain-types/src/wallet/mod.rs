mod address;
pub(crate) mod private_key;

pub use address::Address;
use ed25519_dalek::SigningKey;
use private_key::PrivateKey;
use rand::rngs::OsRng;

use crate::transaction::{Transaction, TransactionConstructor};

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
        assert!(tx.is_validate());
    }
}
