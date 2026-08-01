import { useState, useRef } from 'react'
import axios from 'axios'
import './App.css'

const API_URL = 'http://127.0.0.1:8000'

function App() {
  const [file, setFile] = useState(null)
  const [uploading, setUploading] = useState(false)
  const [uploadStatus, setUploadStatus] = useState(null) // { text, ok }
  const [messages, setMessages] = useState([])
  const [question, setQuestion] = useState('')
  const [loading, setLoading] = useState(false)
  const fileInputRef = useRef(null)

  const handleFileChange = (e) => {
    setFile(e.target.files[0])
    setUploadStatus(null)
  }

  const handleUpload = async () => {
    if (!file) return
    setUploading(true)
    setUploadStatus(null)

    const formData = new FormData()
    formData.append('file', file)

    try {
      const res = await axios.post(`${API_URL}/upload`, formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
      })
      setUploadStatus({
        text: `${res.data.filename} — ${res.data.chunks_created} chunks indexed`,
        ok: true,
      })
    } catch (err) {
      setUploadStatus({
        text: err.response?.data?.detail || err.message,
        ok: false,
      })
    } finally {
      setUploading(false)
    }
  }

  const handleAsk = async () => {
    if (!question.trim()) return
    const userMessage = { role: 'user', content: question }
    setMessages((prev) => [...prev, userMessage])
    setQuestion('')
    setLoading(true)

    try {
      const res = await axios.post(`${API_URL}/chat`, { question: userMessage.content })
      setMessages((prev) => [
        ...prev,
        { role: 'bot', content: res.data.answer, sources: res.data.sources },
      ])
    } catch (err) {
      setMessages((prev) => [
        ...prev,
        { role: 'bot', content: err.response?.data?.detail || err.message, isError: true },
      ])
    } finally {
      setLoading(false)
    }
  }

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleAsk()
    }
  }

  return (
    <div className="app">
      <header className="header">
        <div className="header-mark">DT</div>
        <div>
          <h1>DocTalk</h1>
          <p>Answers pulled from your document — never invented</p>
        </div>
      </header>

      <div className="doc-card">
        <div className="doc-card-label">Source document</div>
        <div className="doc-card-body">
          <input
            id="file-input"
            type="file"
            accept="application/pdf"
            onChange={handleFileChange}
            ref={fileInputRef}
            className="file-input-hidden"
          />
          <label htmlFor="file-input" className="file-label">
            {file ? file.name : 'Choose a PDF'}
          </label>
          <button onClick={handleUpload} disabled={!file || uploading} className="upload-btn">
            {uploading ? 'Indexing…' : 'Upload'}
          </button>
        </div>
        {uploadStatus && (
          <div className={`upload-status ${uploadStatus.ok ? 'ok' : 'err'}`}>
            {uploadStatus.text}
          </div>
        )}
      </div>

      <div className="chat-window">
        {messages.length === 0 && (
          <div className="empty-state">
            Upload a document above, then ask it something.
          </div>
        )}
        {messages.map((msg, i) => (
          <div key={i} className={`message ${msg.role} ${msg.isError ? 'error' : ''}`}>
            <div className="message-content">{msg.content}</div>
            {msg.sources && msg.sources.length > 0 && (
              <details className="sources">
                <summary>{msg.sources.length} source{msg.sources.length > 1 ? 's' : ''}</summary>
                <div className="source-list">
                  {msg.sources.map((src, j) => (
                    <div key={j} className="source-chunk">{src}</div>
                  ))}
                </div>
              </details>
            )}
          </div>
        ))}
        {loading && (
          <div className="message bot">
            <div className="thinking">
              <span></span><span></span><span></span>
            </div>
          </div>
        )}
      </div>

      <div className="input-section">
        <textarea
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="Ask a question about your document…"
          rows={1}
        />
        <button onClick={handleAsk} disabled={loading || !question.trim()}>
          Ask
        </button>
      </div>
    </div>
  )
}

export default App