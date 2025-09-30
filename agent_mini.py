import sys, time

print("✅ OM1 session ready (agent_mini)")
print("Ketik 'exit' untuk keluar.\n")

def respond(msg: str) -> str:
    msg = msg.strip().lower()
    if not msg:
        return "Aku menunggu input."
    if msg in {"exit", "quit"}:
        return "Sampai jumpa."
    # aturan sederhana (placeholder sebelum pakai LLM/ROS/Gazebo)
    if "btc" in msg or "bitcoin" in msg:
        return "Sinyal placeholder: pantau volatilitas; ini hanya demo tanpa data pasar."
    if "nodepay" in msg or "nc" in msg:
        return "Sinyal placeholder: staking akses produk; ini hanya demo lokal."
    return f"Echo (demo): {msg}"

while True:
    try:
        user = input("YOU> ")
        reply = respond(user)
        print("AGENT>", reply)
        if user.strip().lower() in {"exit","quit"}:
            break
    except KeyboardInterrupt:
        print("\nAGENT> Dihentikan.")
        break
