import { useState } from "react";
import "./App.css";

function App() {
  const [mode, setMode] = useState("chat");
  const [question, setQuestion] = useState("");
  const [answer, setAnswer] = useState(null);
  const [sources, setSources] = useState([]);
  const [billA, setBillA] = useState("");
  const [billB, setBillB] = useState("");
  const [conflict, setConflict] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  async function askQuestion() {
    if (!question.trim()) {
      return;
    }

    setLoading(true);
    setError("");
    setAnswer(null);
    setSources([]);

    try {
      const response = await fetch(
        "http://127.0.0.1:8000/api/chat",
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            question: question,
          }),
        }
      );

      if (!response.ok) {
        throw new Error("Failed to get answer");
      }

      const data = await response.json();

      setAnswer(data.answer);
      setSources(data.sources || []);

    } catch (err) {
      setError(err.message);

    } finally {
      setLoading(false);
    }
  }


  async function analyzeConflict() {
    if (!billA.trim() || !billB.trim()) {
      return;
    }

    setLoading(true);
    setError("");
    setConflict(null);

    try {
      const response = await fetch(
        "http://127.0.0.1:8000/api/conflict",
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            bill_a: billA,
            bill_b: billB,
          }),
        }
      );

      if (!response.ok) {
        throw new Error("Failed to analyze conflict");
      }

      const data = await response.json();

      setConflict(data);

    } catch (err) {
      setError(err.message);

    } finally {
      setLoading(false);
    }
  }


  return (
    <div className="app">

      <header className="header">
        <h1>LegalSpace</h1>

        <p>
          Retrieval-grounded constitutional analysis
        </p>
      </header>


      <div className="tabs">

        <button
          className={mode === "chat" ? "active" : ""}
          onClick={() => setMode("chat")}
        >
          Constitutional Q&A
        </button>

        <button
          className={mode === "conflict" ? "active" : ""}
          onClick={() => setMode("conflict")}
        >
          Bill Conflict Analysis
        </button>

      </div>


      {mode === "chat" && (

        <main className="panel">

          <h2>Ask a constitutional question</h2>

          <textarea
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            placeholder="What protects equality before the law?"
          />

          <button
            className="primary"
            onClick={askQuestion}
            disabled={loading}
          >
            {loading ? "Analyzing..." : "Ask LegalSpace"}
          </button>


          {error && (
            <div className="error">
              {error}
            </div>
          )}


          {answer && (

            <section className="result">

              <h2>Answer</h2>

              <p className="answer">
                {answer}
              </p>


              {sources.length > 0 && (

                <div className="sources">

                  <h3>Sources</h3>

                  {sources.map((source, index) => (

                    <div
                      className="source"
                      key={index}
                    >

                      <strong>
                        {source.article}
                      </strong>

                      <p>
                        {source.text}
                      </p>

                      <small>
                        Rerank score:{" "}
                        {source.rerank_score?.toFixed(4)}
                      </small>

                    </div>

                  ))}

                </div>

              )}

            </section>

          )}

        </main>

      )}


      {mode === "conflict" && (

        <main className="panel">

          <h2>Cross-Bill Conflict Analysis</h2>

          <div className="bill-grid">

            <div>

              <h3>Bill A</h3>

              <textarea
                value={billA}
                onChange={(e) => setBillA(e.target.value)}
                placeholder="Enter the first legislative provision..."
              />

            </div>


            <div>

              <h3>Bill B</h3>

              <textarea
                value={billB}
                onChange={(e) => setBillB(e.target.value)}
                placeholder="Enter the second legislative provision..."
              />

            </div>

          </div>


          <button
            className="primary"
            onClick={analyzeConflict}
            disabled={loading}
          >
            {loading
              ? "Analyzing..."
              : "Analyze Conflict"}
          </button>


          {error && (
            <div className="error">
              {error}
            </div>
          )}


          {conflict && (

            <section className="result">

              <h2>Analysis</h2>

              <div className="conflict-status">

                <strong>
                  {conflict.conflict_type}
                </strong>

                <span>
                  Confidence: {conflict.confidence}
                </span>

              </div>


              <h3>Explanation</h3>

              <p>
                {conflict.conflict_explanation}
              </p>


              {conflict.relevant_articles?.length > 0 && (

                <>
                  <h3>Relevant Articles</h3>

                  <ul>
                    {conflict.relevant_articles.map(
                      (article, index) => (
                        <li key={index}>
                          {article}
                        </li>
                      )
                    )}
                  </ul>
                </>

              )}


              {conflict.supporting_evidence?.length > 0 && (

                <>
                  <h3>Supporting Evidence</h3>

                  {conflict.supporting_evidence.map(
                    (evidence, index) => (
                      <p key={index}>
                        {evidence}
                      </p>
                    )
                  )}
                </>
              )}
            </section>
          )}
        </main>
      )}
    </div>
  );
}

export default App;
