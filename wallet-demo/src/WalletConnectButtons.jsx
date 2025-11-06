import React, { useState } from "react";
import {
  useConnect,
  useAccount,
  useDisconnect,
  useBalance,
  useChainId,
  useSwitchChain,
  useSendTransaction,
  useSignMessage,
} from "wagmi";
import { parseEther } from "viem";

export default function WalletConnectButtons() {
  const { connectors, connect, error, isPending, pendingConnector } = useConnect();
  const { address, isConnected } = useAccount();
  const { disconnect } = useDisconnect();
  const chainId = useChainId();
  const { switchChain } = useSwitchChain();
  const { data: balance } = useBalance({ address });
  const { sendTransaction } = useSendTransaction();
  const { signMessageAsync } = useSignMessage();
  const [loading, setLoading] = useState(false);

  // ---- CONNECTED UI ----
  if (isConnected)
    return (
      <div
        style={{
          display: "flex",
          flexDirection: "column",
          alignItems: "center",
          background: "#000",
          color: "#0f0",
          padding: "2rem",
          borderRadius: "15px",
          minWidth: "420px",
        }}
      >
        <h2>‚úÖ Connected</h2>
        <p>Address: {address}</p>
        <p>Network ID: {chainId}</p>
        <p>Balance: {balance ? `${balance.formatted} ${balance.symbol}` : "Loading..."}</p>

        <div style={{ marginTop: "1rem" }}>
          <button
            onClick={async () => {
              setLoading(true);
              try {
                await signMessageAsync({ message: "Hello from OM1 Wallet Ì±ã" });
                alert("Message signed successfully!");
              } catch (err) {
                console.error(err);
              }
              setLoading(false);
            }}
            style={btn}
          >
            ‚úçÔ∏è Sign Message
          </button>

          <button
            onClick={() =>
              sendTransaction({
                to: address,
                value: parseEther("0.001"),
              })
            }
            style={btn}
          >
            Ì≤∏ Send 0.001 ETH
          </button>

          <div style={{ marginTop: "10px" }}>
            <h4>Ì¥Å Switch Network</h4>
            {switchChain && (
              <>
                <button onClick={() => switchChain({ chainId: 11155111 })} style={smallBtn}>
                  Sepolia
                </button>
                <button onClick={() => switchChain({ chainId: 84532 })} style={smallBtn}>
                  Base Sepolia
                </button>
                <button onClick={() => switchChain({ chainId: 59141 })} style={smallBtn}>
                  Linea Goerli
                </button>
                <button onClick={() => switchChain({ chainId: 1 })} style={smallBtn}>
                  Ethereum
                </button>
              </>
            )}
          </div>

          <button onClick={() => disconnect()} style={disconnectBtn}>
            ‚ùå Disconnect
          </button>
        </div>

        {loading && <p>‚è≥ Please wait...</p>}
      </div>
    );

  // ---- CONNECT BUTTONS UI ----
  return (
    <div style={{ textAlign: "center", marginTop: "40px" }}>
      <h3>Ì≤° Connect Wallet</h3>
      {connectors.map((connector) => (
        <button
          disabled={!connector.ready}
          key={connector.uid}
          onClick={() => connect({ connector })}
          style={btn}
        >
          {connector.name}
          {isPending && pendingConnector?.id === connector.id && " (connecting...)"}
        </button>
      ))}
      {error && <p style={{ color: "red" }}>{error.message}</p>}
    </div>
  );
}

// ---- Button Styles ----
const btn = {
  background: "#0f0",
  color: "#000",
  border: "none",
  borderRadius: "10px",
  padding: "10px 15px",
  margin: "8px",
  fontWeight: "bold",
  cursor: "pointer",
  width: "220px",
};

const smallBtn = {
  ...btn,
  background: "#333",
  color: "#0f0",
  width: "140px",
  margin: "4px",
};

const disconnectBtn = {
  ...btn,
  background: "red",
  color: "white",
  marginTop: "20px",
};
