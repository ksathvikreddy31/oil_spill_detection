import React, { useState } from "react";
import axios from "axios";

function App() {
  const [file, setFile] = useState(null);
  const [fileName, setFileName] = useState("");
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const handleFileChange = (e) => {
    const selectedFile = e.target.files[0];
    if (selectedFile) {
      setFile(selectedFile);
      setFileName(selectedFile.name);
      setResult(null);
      setError("");
    }
  };

  const uploadImage = async () => {
    if (!file) {
      setError("Please select an image file first.");
      return;
    }

    setLoading(true);
    setError("");
    
    const formData = new FormData();
    formData.append("file", file);

    try {
      const res = await axios.post("http://localhost:8000/predict", formData);
      setResult(res.data);
    } catch (err) {
      setError("Failed to connect to the analysis engine. Ensure the backend is running.");
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="app-container">
      <header className="header">
        <h1>Oil Spill Detection System</h1>
        <p>Satellite Imagery Intelligence & Neural Segmentation Engine</p>
      </header>

      <div className="upload-card">
        <div className="file-input-wrapper">
          <label htmlFor="file-upload" className="custom-file-upload">
            {fileName || "Click to select Satellite Image"}
          </label>
          <input 
            id="file-upload" 
            type="file" 
            onChange={handleFileChange}
            accept="image/*"
          />
        </div>
        
        <div style={{ marginTop: "10px" }}>
          <button 
            className="upload-btn" 
            onClick={uploadImage} 
            disabled={loading || !file}
          >
            {loading ? (
              <>
                <span className="loader"></span>
                Processing...
              </>
            ) : "Start Analysis"}
          </button>
        </div>

        {error && <p style={{ color: "var(--accent-red)", marginTop: "15px", fontSize: "0.9rem" }}>{error}</p>}
      </div>

      {result && (
        <div className="fade-in">
          <div className={`status-banner ${result.decision.includes("NO") ? "status-clear" : "status-detected"}`}>
            {result.decision}
          </div>

          <div className="results-grid">
            <div className="result-card">
              <h3>Original Input</h3>
              <img src={result.original} className="result-image" alt="Original Satellite View" />
            </div>

            <div className="result-card">
              <h3>DL Segmentations</h3>
              <img src={result.dl_pred} className="result-image" alt="Deep Learning Mask" />
            </div>

            <div className="result-card">
              <h3>Ensemble Results</h3>
              <img src={result.final_pred} className="result-image" alt="Final Machine Learning Prediction" />
            </div>
          </div>

          <div className="data-section">
            <div className="data-card">
              <h2>📊 Performance Metrics</h2>
              <table>
                <thead>
                  <tr>
                    <th>Metric</th>
                    <th>Score</th>
                  </tr>
                </thead>
                <tbody>
                  {result.metrics.map((m, i) => (
                    <tr key={i}>
                      <td>{m.Metric}</td>
                      <td>{(parseFloat(m.Value) * 100).toFixed(2)}%</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>

            <div className="data-card">
              <h2>🧮 Confusion Matrix</h2>
              <table className="confusion-matrix">
                <thead>
                  <tr className="cm-header">
                    <th></th>
                    <th style={{ textAlign: "center" }}>No Oil (Pred)</th>
                    <th style={{ textAlign: "center" }}>Oil (Pred)</th>
                  </tr>
                </thead>
                <tbody>
                  <tr>
                    <td className="cm-label">Actual Clear</td>
                    <td className="cm-value" style={{ color: "var(--accent-cyan)" }}>{result.cm[0][0]}</td>
                    <td className="cm-value" style={{ color: "var(--accent-red)" }}>{result.cm[0][1]}</td>
                  </tr>
                  <tr>
                    <td className="cm-label">Actual Spill</td>
                    <td className="cm-value" style={{ color: "var(--accent-orange)" }}>{result.cm[1][0]}</td>
                    <td className="cm-value" style={{ color: "var(--accent-green)" }}>{result.cm[1][1]}</td>
                  </tr>
                </tbody>
              </table>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

export default App;