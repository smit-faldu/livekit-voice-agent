import { useEffect, useRef, useState } from 'react'
import {
  BarVisualizer,
  RoomAudioRenderer,
  SessionProvider,
  TrackToggle,
  useAgent,
  useSession,
  useSessionMessages,
} from '@livekit/components-react'
import { Track, TokenSource } from 'livekit-client'

const KEY_STORAGE = 'assistant-access-key'

// localStorage can throw (private mode, blocked storage), so never let it break the app.
const loadKey = () => { try { return localStorage.getItem(KEY_STORAGE) ?? '' } catch { return '' } }
const saveKey = (k: string) => { try { localStorage.setItem(KEY_STORAGE, k) } catch { /* ignore */ } }

// 1. Where to get a token: our FastAPI endpoint on the same server, in LiveKit's standard
//    format. It needs the access key (APP_ACCESS_KEY in .env.local) as a Bearer header.
//    `custom` keeps ONE stable token source that reads the current key at fetch time.
// Set from storage at load: useSession prefetches a token on page load (prepareConnection).
let accessKeyForFetch = loadKey()
const tokenSource = TokenSource.custom(async (options) => {
  const res = await fetch('/api/token', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${accessKeyForFetch}` },
    // Ask the server to dispatch our agent into the new room.
    body: JSON.stringify({ room_config: { agents: [{ agent_name: options.agentName }] } }),
  })
  if (!res.ok) throw new Error(res.status === 401 ? 'Wrong access key.' : `Token request failed (${res.status})`)
  const { server_url, participant_token } = await res.json()
  return { serverUrl: server_url, participantToken: participant_token }
})

export default function App() {
  const [accessKey, setAccessKey] = useState(loadKey)
  // 2. One session = token fetch + room connect + agent dispatch + cleanup.
  const session = useSession(tokenSource, { agentName: 'assistant' })
  const [starting, setStarting] = useState(false)
  const [error, setError] = useState('')

  const start = async () => {
    setStarting(true)
    setError('')
    try {
      saveKey(accessKey)
      // Called from a click, so the browser allows mic access and audio playback.
      await session.start({ tracks: { microphone: { enabled: true } } })
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setStarting(false)
    }
  }

  return (
    <main className="app">
      <header>
        <h1>Local Voice Assistant</h1>
        <p className="sub">Whisper · Gemini · Piper, over your own LiveKit server</p>
      </header>

      {session.isConnected ? (
        // 3. SessionProvider gives every child the room + session through React context.
        <SessionProvider session={session}>
          <Conversation onEnd={() => session.end()} />
          <RoomAudioRenderer /> {/* plays the agent's voice */}
        </SessionProvider>
      ) : (
        <form className="start" onSubmit={(e) => { e.preventDefault(); start() }}>
          <label htmlFor="key">Access key</label>
          <input
            id="key"
            type="password"
            value={accessKey}
            onChange={(e) => { setAccessKey(e.target.value); accessKeyForFetch = e.target.value }}
            placeholder="APP_ACCESS_KEY from .env.local"
            autoComplete="current-password"
          />
          <button className="primary" type="submit" disabled={starting || !accessKey}>
            {starting ? 'Connecting…' : 'Start conversation'}
          </button>
          {error && <p className="error" role="alert">{error}</p>}
        </form>
      )}
    </main>
  )
}

function Conversation({ onEnd }: { onEnd: () => void }) {
  // 4. Agent state comes from the agent's `lk.agent.state` attribute.
  const agent = useAgent()
  // 5. Transcripts (spoken) and chat (typed) arrive as one message list over LiveKit text streams.
  const { messages, send, isSending } = useSessionMessages()
  const [draft, setDraft] = useState('')
  const listRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    listRef.current?.scrollTo({ top: listRef.current.scrollHeight })
  }, [messages])

  const submit = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!draft.trim()) return
    await send(draft.trim()) // agent answers typed text too (and speaks the reply)
    setDraft('')
  }

  return (
    <section className="conversation">
      <div className="status">
        <BarVisualizer state={agent.state} track={agent.microphoneTrack} barCount={7} className="viz" />
        <span className={`badge ${agent.state}`}>{agent.state}</span>
      </div>

      <div className="messages" ref={listRef} aria-live="polite">
        {messages.length === 0 && <p className="hint">Say something, or type below.</p>}
        {messages.map((m) => {
          const mine = m.type === 'userTranscript' || (m.type !== 'agentTranscript' && m.from?.isLocal)
          return (
            <div key={m.id} className={`msg ${mine ? 'user' : 'agent'}`}>
              <span className="who">{mine ? 'You' : 'Assistant'}</span>
              {m.message}
            </div>
          )
        })}
      </div>

      <form className="composer" onSubmit={submit}>
        <input
          value={draft}
          onChange={(e) => setDraft(e.target.value)}
          placeholder="Type a message…"
          aria-label="Message"
        />
        <button type="submit" disabled={isSending || !draft.trim()}>Send</button>
      </form>

      <div className="controls">
        <TrackToggle source={Track.Source.Microphone}>Mic</TrackToggle>
        <button className="danger" onClick={onEnd}>End</button>
      </div>
    </section>
  )
}
