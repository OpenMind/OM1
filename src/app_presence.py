# app_presence.py (demo script)
import asyncio
from providers.face_presence_provider import FacePresenceProvider
from inputs.plugins.face_presence_input import FacePresenceInput

async def main():
    provider = FacePresenceProvider(
        base_url="http://127.0.0.1:6793",  
        recent_sec=2.0,
        fps=5.0,
        capacity=300,
    )
    inp = FacePresenceInput(provider, poll_interval_s=0.2)
    await inp.start()


    for _ in range(5):
        print(inp.formatted_latest_buffer(history_sec=2.0))
        await asyncio.sleep(1.0)


    latest = inp.get_latest()
    if latest:
        prompt = f"Camera presence update: {latest.text}. React accordingly."
        print("LLM prompt:", prompt)

    await inp.stop()
    provider.stop()

if __name__ == "__main__":
    asyncio.run(main())
