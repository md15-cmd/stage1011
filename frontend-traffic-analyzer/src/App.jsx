import React, { useState, useEffect } from 'react';
import { Upload, FileText, Brain, Shield, Network, Activity, Eye, TrendingUp, AlertTriangle, CheckCircle, XCircle } from 'lucide-react';

const TrafficClassifier = () => {
  const [file, setFile] = useState(null);
  const [results, setResults] = useState(null);
  const [loading, setLoading] = useState(false);
  const [dragOver, setDragOver] = useState(false);
  const [error, setError] = useState(null);
  const [backendStatus, setBackendStatus] = useState('checking');
  const [stats, setStats] = useState({
    totalPredictions: 0,
    accuracy: 94.2, // Valeur par défaut du modèle
    threatLevel: 'En attente...'
  });

  const API_BASE_URL = 'http://localhost:5000';

  // Vérifier l'état du backend au chargement
  useEffect(() => {
    checkBackendHealth();
    fetchStats();
  }, []);

  const checkBackendHealth = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/health`);
      if (response.ok) {
        const health = await response.json();
        setBackendStatus(health.model_loaded ? 'online' : 'model_error');
      } else {
        setBackendStatus('offline');
      }
    } catch (error) {
      setBackendStatus('offline');
    }
  };

  const fetchStats = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/stats`);
      if (response.ok) {
        const statsData = await response.json();
        setStats(prev => ({
          ...prev,
          totalPredictions: statsData.totalPredictions || prev.totalPredictions,
          threatLevel: statsData.threatLevel || prev.threatLevel
        }));
      }
    } catch (error) {
      console.log('Could not fetch stats from backend, using defaults');
      // Garder les valeurs par défaut si le backend n'est pas disponible
    }
  };

  const handleFileUpload = async (uploadedFile) => {
    if (backendStatus !== 'online') {
      setError('Le backend n\'est pas disponible. Veuillez vérifier que le serveur Flask est en cours d\'exécution.');
      return;
    }

    setFile(uploadedFile);
    setLoading(true);
    setError(null);
    setResults(null);

    const formData = new FormData();
    formData.append('file', uploadedFile);

    try {
      const response = await fetch(`${API_BASE_URL}/predict`, {
        method: 'POST',
        body: formData,
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.error || 'Erreur lors de la prédiction');
      }

      setResults(data);
      
      // Mettre à jour les statistiques avec les nouvelles données
      if (data.stats) {
        setStats(prev => ({
          totalPredictions: prev.totalPredictions + data.stats.total_samples,
          accuracy: prev.accuracy, // Garder l'ancienne précision du modèle
          threatLevel: data.stats.threat_level || prev.threatLevel
        }));
      }

    } catch (error) {
      setError(error.message);
      console.error('Erreur:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setDragOver(false);
    const droppedFile = e.dataTransfer.files[0];
    if (droppedFile && droppedFile.name.endsWith('.csv')) {
      handleFileUpload(droppedFile);
    } else {
      setError('Veuillez sélectionner un fichier CSV valide.');
    }
  };

  const handleFileSelect = (e) => {
    const selectedFile = e.target.files[0];
    if (selectedFile) {
      handleFileUpload(selectedFile);
    }
  };

  const getRiskColor = (risk) => {
    switch(risk?.toLowerCase()) {
      case 'high': return 'text-red-600 bg-red-50';
      case 'medium': return 'text-yellow-600 bg-yellow-50';
      case 'low': return 'text-green-600 bg-green-50';
      default: return 'text-gray-600 bg-gray-50';
    }
  };

  const getConfidenceColor = (confidence) => {
    if (confidence >= 0.9) return 'text-green-600';
    if (confidence >= 0.7) return 'text-yellow-600';
    return 'text-red-600';
  };

  const getStatusIcon = () => {
    switch(backendStatus) {
      case 'online': return <CheckCircle className="w-4 h-4 text-green-400" />;
      case 'offline': return <XCircle className="w-4 h-4 text-red-400" />;
      case 'model_error': return <AlertTriangle className="w-4 h-4 text-yellow-400" />;
      default: return <div className="w-4 h-4 animate-spin rounded-full border-2 border-blue-400 border-t-transparent" />;
    }
  };

  const getStatusText = () => {
    switch(backendStatus) {
      case 'online': return 'Backend connecté';
      case 'offline': return 'Backend déconnecté';
      case 'model_error': return 'Erreur modèle';
      default: return 'Vérification...';
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-blue-900 to-indigo-900">
      {/* Header */}
      <div className="bg-white/10 backdrop-blur-md border-b border-white/20">
        <div className="max-w-7xl mx-auto px-6 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-3">
              <div className="p-2 bg-blue-500 rounded-lg">
                <Shield className="w-6 h-6 text-white" />
              </div>
              <div>
                <h1 className="text-2xl font-bold text-white">Traffic Classifier</h1>
                <p className="text-blue-200">Intelligence Artificielle pour l'Analyse de Trafic Réseau</p>
              </div>
            </div>
            <div className="flex items-center space-x-2 text-sm">
              {getStatusIcon()}
              <span className="text-gray-300">{getStatusText()}</span>
            </div>
          </div>
        </div>
      </div>

      {/* Dashboard Stats */}
      <div className="max-w-7xl mx-auto px-6 py-8">
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">
          <div className="bg-white/10 backdrop-blur-md rounded-xl p-6 border border-white/20">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-blue-200 text-sm">Total Prédictions</p>
                <p className="text-2xl font-bold text-white">{stats.totalPredictions.toLocaleString()}</p>
              </div>
              <TrendingUp className="w-8 h-8 text-blue-400" />
            </div>
          </div>
          <div className="bg-white/10 backdrop-blur-md rounded-xl p-6 border border-white/20">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-green-200 text-sm">Précision du Modèle</p>
                <p className="text-2xl font-bold text-white">{stats.accuracy}%</p>
              </div>
              <Brain className="w-8 h-8 text-green-400" />
            </div>
          </div>
          <div className="bg-white/10 backdrop-blur-md rounded-xl p-6 border border-white/20">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-orange-200 text-sm">Niveau de Menace</p>
                <p className="text-2xl font-bold text-white">{stats.threatLevel}</p>
              </div>
              <AlertTriangle className="w-8 h-8 text-orange-400" />
            </div>
          </div>
        </div>

        {/* Error Message */}
        {error && (
          <div className="mb-6 bg-red-500/20 border border-red-500/50 rounded-xl p-4">
            <div className="flex items-center space-x-2">
              <AlertTriangle className="w-5 h-5 text-red-400" />
              <p className="text-red-200">{error}</p>
            </div>
          </div>
        )}

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          {/* Upload Section */}
          <div className="lg:col-span-1">
            <div className="bg-white/10 backdrop-blur-md rounded-xl p-6 border border-white/20">
              <h2 className="text-xl font-semibold text-white mb-4 flex items-center">
                <Upload className="w-5 h-5 mr-2" />
                Charger les Données
              </h2>
              
              <div
                className={`border-2 border-dashed rounded-lg p-8 text-center transition-all duration-300 ${
                  dragOver 
                    ? 'border-blue-400 bg-blue-400/10' 
                    : backendStatus === 'online'
                    ? 'border-gray-400 hover:border-blue-400'
                    : 'border-gray-600 cursor-not-allowed opacity-50'
                }`}
                onDrop={backendStatus === 'online' ? handleDrop : undefined}
                onDragOver={(e) => { 
                  e.preventDefault(); 
                  if (backendStatus === 'online') setDragOver(true); 
                }}
                onDragLeave={() => setDragOver(false)}
              >
                {loading ? (
                  <div className="flex flex-col items-center">
                    <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-400 mb-4"></div>
                    <p className="text-white">Analyse en cours...</p>
                    <p className="text-gray-300 text-sm mt-2">Cela peut prendre quelques secondes</p>
                  </div>
                ) : (
                  <div>
                    <FileText className="w-12 h-12 text-blue-400 mx-auto mb-4" />
                    <p className="text-white mb-2">Glissez votre fichier CSV ici</p>
                    <p className="text-gray-300 text-sm mb-4">ou</p>
                    <label className={`inline-flex items-center px-4 py-2 rounded-lg transition-colors ${
                      backendStatus === 'online'
                        ? 'bg-blue-600 text-white hover:bg-blue-700 cursor-pointer'
                        : 'bg-gray-600 text-gray-300 cursor-not-allowed'
                    }`}>
                      <Upload className="w-4 h-4 mr-2" />
                      Sélectionner un fichier
                      <input
                        type="file"
                        className="hidden"
                        accept=".csv"
                        onChange={handleFileSelect}
                        disabled={backendStatus !== 'online'}
                      />
                    </label>
                  </div>
                )}
              </div>

              {file && (
                <div className="mt-4 p-3 bg-blue-500/20 rounded-lg">
                  <p className="text-white text-sm">
                    <strong>Fichier sélectionné:</strong> {file.name}
                  </p>
                  <p className="text-gray-300 text-xs mt-1">
                    Taille: {(file.size / 1024).toFixed(1)} KB
                  </p>
                </div>
              )}

              {backendStatus !== 'online' && (
                <div className="mt-4 p-3 bg-yellow-500/20 rounded-lg">
                  <p className="text-yellow-200 text-sm">
                    <strong>Note:</strong> Assurez-vous que le serveur Flask est en cours d'exécution sur le port 5000.
                  </p>
                  <button 
                    onClick={checkBackendHealth}
                    className="mt-2 text-xs bg-yellow-600 hover:bg-yellow-700 px-2 py-1 rounded text-white transition-colors"
                  >
                    Vérifier à nouveau
                  </button>
                </div>
              )}
            </div>
          </div>

          {/* Results Section */}
          <div className="lg:col-span-2">
            {results ? (
              <div className="space-y-6">
                {/* Processing Stats */}
                {results.stats && (
                  <div className="bg-white/10 backdrop-blur-md rounded-xl p-6 border border-white/20">
                    <h3 className="text-lg font-semibold text-white mb-4">Statistiques de Traitement</h3>
                    <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                      <div className="text-center p-4 bg-white/5 rounded-lg">
                        <p className="text-2xl font-bold text-white">{results.stats.total_samples}</p>
                        <p className="text-sm text-gray-300">Échantillons</p>
                      </div>
                      <div className="text-center p-4 bg-white/5 rounded-lg">
                        <p className="text-2xl font-bold text-white">{results.stats.processed_samples}</p>
                        <p className="text-sm text-gray-300">Traités</p>
                      </div>
                      <div className="text-center p-4 bg-white/5 rounded-lg">
                        <p className="text-2xl font-bold text-white">{results.stats.high_risk_count}</p>
                        <p className="text-sm text-gray-300">Haut Risque</p>
                      </div>
                      <div className="text-center p-4 bg-white/5 rounded-lg">
                        <p className="text-2xl font-bold text-white">{results.stats.threat_level}</p>
                        <p className="text-sm text-gray-300">Menace</p>
                      </div>
                    </div>
                  </div>
                )}

                {/* Summary Cards */}
                <div className="bg-white/10 backdrop-blur-md rounded-xl p-6 border border-white/20">
                  <h3 className="text-lg font-semibold text-white mb-4 flex items-center">
                    <Activity className="w-5 h-5 mr-2" />
                    Résumé des Classifications
                  </h3>
                  <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                    {Object.entries(results.summary).map(([type, count]) => (
                      <div key={type} className="text-center p-4 bg-white/5 rounded-lg">
                        <p className="text-2xl font-bold text-white">{count}</p>
                        <p className="text-sm text-gray-300">{type}</p>
                      </div>
                    ))}
                  </div>
                </div>

                {/* Detailed Results */}
                <div className="bg-white/10 backdrop-blur-md rounded-xl p-6 border border-white/20">
                  <h3 className="text-lg font-semibold text-white mb-4 flex items-center">
                    <Eye className="w-5 h-5 mr-2" />
                    Détails des Prédictions
                    {results.predictions.length >= 100 && (
                      <span className="ml-2 text-sm text-yellow-300">(100 premiers résultats)</span>
                    )}
                  </h3>
                  <div className="overflow-x-auto">
                    <table className="w-full text-sm">
                      <thead>
                        <tr className="border-b border-white/20">
                          <th className="text-left py-3 px-4 text-gray-300">Flow ID</th>
                          <th className="text-left py-3 px-4 text-gray-300">Prédiction</th>
                          <th className="text-left py-3 px-4 text-gray-300">Confiance</th>
                          <th className="text-left py-3 px-4 text-gray-300">Niveau de Risque</th>
                        </tr>
                      </thead>
                      <tbody>
                        {results.predictions.map((result) => (
                          <tr key={result.id} className="border-b border-white/10 hover:bg-white/5">
                            <td className="py-3 px-4 text-white font-mono">{result.flow}</td>
                            <td className="py-3 px-4">
                              <span className="inline-flex items-center px-2 py-1 rounded-full text-xs font-medium bg-blue-500/20 text-blue-300">
                                {result.prediction}
                              </span>
                            </td>
                            <td className={`py-3 px-4 font-semibold ${getConfidenceColor(result.confidence)}`}>
                              {(result.confidence * 100).toFixed(1)}%
                            </td>
                            <td className="py-3 px-4">
                              <span className={`inline-flex items-center px-2 py-1 rounded-full text-xs font-medium ${getRiskColor(result.risk)}`}>
                                {result.risk}
                              </span>
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>
              </div>
            ) : (
              <div className="bg-white/10 backdrop-blur-md rounded-xl p-12 border border-white/20 text-center">
                <Network className="w-16 h-16 text-gray-400 mx-auto mb-4" />
                <h3 className="text-xl font-semibold text-white mb-2">Prêt pour l'Analyse</h3>
                <p className="text-gray-300">
                  Chargez un fichier CSV contenant les données de trafic réseau pour commencer l'analyse avec notre modèle LSTM.
                </p>
                {backendStatus === 'online' && (
                  <p className="text-green-300 text-sm mt-2">✓ Backend connecté et modèle chargé</p>
                )}
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default TrafficClassifier;