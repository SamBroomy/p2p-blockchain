/// Network configuration
#[derive(Debug, Clone)]
pub struct NetworkConfig {
    /// TCP listen address (e.g., "/ip4/0.0.0.0/tcp/0")
    pub tcp_listen_addr: String,

    /// Optional: Explicit peers to dial on startup
    pub bootstrap_peers: Vec<String>,
}

impl Default for NetworkConfig {
    fn default() -> Self {
        Self {
            tcp_listen_addr: "/ip4/0.0.0.0/tcp/0".to_string(),
            bootstrap_peers: Vec::new(),
        }
    }
}
