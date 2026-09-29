"""Phase 1: agent joins a room and logs who is there.

No AI yet. Goal: see the transport layer work end to end.
Run: uv run steps/01_join_room.py dev
"""

import logging

from dotenv import load_dotenv
from livekit import agents, rtc
from livekit.agents import AgentServer, JobContext

load_dotenv(".env.local")
logger = logging.getLogger("phase1")

server = AgentServer()


# agent_name => explicit dispatch: the agent joins only rooms whose token asks for "assistant".
@server.rtc_session(agent_name="assistant")
async def entrypoint(ctx: JobContext):
    logger.info("job received for room %s", ctx.room.name)

    @ctx.room.on("participant_connected")
    def on_join(p: rtc.RemoteParticipant):
        logger.info("participant joined: %s (%s)", p.identity, p.name)

    @ctx.room.on("participant_disconnected")
    def on_leave(p: rtc.RemoteParticipant):
        logger.info("participant left: %s", p.identity)

    @ctx.room.on("track_subscribed")
    def on_track(track: rtc.Track, pub: rtc.RemoteTrackPublication, p: rtc.RemoteParticipant):
        # kind 1 = audio, 2 = video; source tells mic vs camera vs screen share
        logger.info("subscribed %s track from %s (source=%s)", rtc.TrackKind.Name(track.kind), p.identity, rtc.TrackSource.Name(pub.source))

    await ctx.connect()  # join the room (WebRTC) and auto-subscribe to tracks
    logger.info("agent connected as %s", ctx.room.local_participant.identity)

    for p in ctx.room.remote_participants.values():  # users who joined before us
        logger.info("already in room: %s", p.identity)
    # Returning keeps the job alive until the room closes.


if __name__ == "__main__":
    agents.cli.run_app(server)
