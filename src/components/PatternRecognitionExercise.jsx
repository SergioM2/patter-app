import React, { useState, useEffect } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ScatterChart, Scatter, ResponsiveContainer } from 'recharts';

const PatternRecognitionExercise = () => {
  const [datasets, setDatasets] = useState({});
  const [results, setResults] = useState({});
  const [currentDataset, setCurrentDataset] = useState('D1');

  // Función para generar números aleatorios con distribución normal
  const normalRandom = (mean, variance) => {
    const u1 = Math.random();
    const u2 = Math.random();
    const z0 = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
    return z0 * Math.sqrt(variance) + mean;
  };

  // Función para generar datos multivariados normales
  const generateMultivariateNormal = (mean, covariance, n) => {
    const data = [];
    
    // Descomposición de Cholesky simplificada para matrices 2x2
    const [[a, b], [c, d]] = covariance;
    const L11 = Math.sqrt(a);
    const L21 = c / L11;
    const L22 = Math.sqrt(d - L21 * L21);
    
    for (let i = 0; i < n; i++) {
      const z1 = normalRandom(0, 1);
      const z2 = normalRandom(0, 1);
      
      const x1 = mean[0] + L11 * z1;
      const x2 = mean[1] + L21 * z1 + L22 * z2;
      
      data.push([x1, x2]);
    }
    
    return data;
  };

  // Configuraciones de los datasets
  const datasetConfigs = {
    D1: {
      means: [[1, 1], [12, 8], [16, 1]],
      covariances: [[[4, 0], [0, 4]], [[4, 0], [0, 4]], [[4, 0], [0, 4]]],
      name: 'D1 (Covarianzas Identidad)'
    },
    D2: {
      means: [[1, 1], [14, 7], [16, 1]],
      covariances: [[[5, 3], [3, 4]], [[5, 3], [3, 4]], [[5, 3], [3, 4]]],
      name: 'D2 (Covarianzas Correlacionadas)'
    },
    D3: {
      means: [[1, 1], [8, 6], [13, 1]],
      covariances: [[[6, 0], [0, 6]], [[6, 0], [0, 6]], [[6, 0], [0, 6]]],
      name: 'D3 (Clases más cercanas)'
    },
    D4: {
      means: [[1, 1], [10, 5], [11, 1]],
      covariances: [[[7, 4], [4, 5]], [[7, 4], [4, 5]], [[7, 4], [4, 5]]],
      name: 'D4 (Muy cercanas + correlación)'
    }
  };

  // Función para calcular la inversa de una matriz 2x2
  const matrixInverse2x2 = (matrix) => {
    const [[a, b], [c, d]] = matrix;
    const det = a * d - b * c;
    return [[d / det, -b / det], [-c / det, a / det]];
  };

  // Función para multiplicar matrices
  const matrixMultiply = (A, B) => {
    const result = [];
    for (let i = 0; i < A.length; i++) {
      result[i] = [];
      for (let j = 0; j < B[0].length; j++) {
        let sum = 0;
        for (let k = 0; k < B.length; k++) {
          sum += A[i][k] * B[k][j];
        }
        result[i][j] = sum;
      }
    }
    return result;
  };

  // Clasificador Bayesiano
  const bayesianClassifier = (point, means, covariances, priors = [1/3, 1/3, 1/3]) => {
    let maxProb = -Infinity;
    let bestClass = 0;

    for (let i = 0; i < 3; i++) {
      const diff = [point[0] - means[i][0], point[1] - means[i][1]];
      const invCov = matrixInverse2x2(covariances[i]);
      
      // Calcular (x - μ)ᵀ Σ⁻¹ (x - μ)
      const temp = matrixMultiply([diff], invCov);
      const mahalanobis = matrixMultiply(temp, [[diff[0]], [diff[1]]])[0][0];
      
      // Log determinante para matrices 2x2
      const det = covariances[i][0][0] * covariances[i][1][1] - covariances[i][0][1] * covariances[i][1][0];
      const logDet = Math.log(det);
      
      // Función discriminante gaussiana
      const discriminant = -0.5 * (mahalanobis + logDet) + Math.log(priors[i]);
      
      if (discriminant > maxProb) {
        maxProb = discriminant;
        bestClass = i;
      }
    }
    return bestClass;
  };

  // Clasificador Euclidiano
  const euclideanClassifier = (point, means) => {
    let minDist = Infinity;
    let bestClass = 0;

    for (let i = 0; i < 3; i++) {
      const dist = Math.pow(point[0] - means[i][0], 2) + Math.pow(point[1] - means[i][1], 2);
      if (dist < minDist) {
        minDist = dist;
        bestClass = i;
      }
    }
    return bestClass;
  };

  // Clasificador de Mahalanobis
  const mahalanobisClassifier = (point, means, covariances) => {
    let minDist = Infinity;
    let bestClass = 0;

    for (let i = 0; i < 3; i++) {
      const diff = [point[0] - means[i][0], point[1] - means[i][1]];
      const invCov = matrixInverse2x2(covariances[i]);
      
      const temp = matrixMultiply([diff], invCov);
      const mahalanobisDist = matrixMultiply(temp, [[diff[0]], [diff[1]]])[0][0];
      
      if (mahalanobisDist < minDist) {
        minDist = mahalanobisDist;
        bestClass = i;
      }
    }
    return bestClass;
  };

  // Función para calcular el error de clasificación
  const calculateError = (predictions, trueLabels) => {
    const errors = predictions.reduce((acc, pred, idx) => {
      return acc + (pred !== trueLabels[idx] ? 1 : 0);
    }, 0);
    return (errors / predictions.length * 100).toFixed(2);
  };

  // Generar datasets y calcular resultados
  useEffect(() => {
    const newDatasets = {};
    const newResults = {};

    Object.keys(datasetConfigs).forEach(datasetKey => {
      const config = datasetConfigs[datasetKey];
      const data = [];
      const labels = [];

      // Generar 1000 puntos (333, 333, 334 por clase)
      const samplesPerClass = [333, 333, 334];
      
      for (let classIdx = 0; classIdx < 3; classIdx++) {
        const classData = generateMultivariateNormal(
          config.means[classIdx],
          config.covariances[classIdx],
          samplesPerClass[classIdx]
        );
        
        classData.forEach(point => {
          data.push(point);
          labels.push(classIdx);
        });
      }

      newDatasets[datasetKey] = {
        data,
        labels,
        config
      };

      // Calcular clasificaciones
      const bayesianPreds = data.map(point => 
        bayesianClassifier(point, config.means, config.covariances)
      );
      const euclideanPreds = data.map(point => 
        euclideanClassifier(point, config.means)
      );
      const mahalanobisPreds = data.map(point => 
        mahalanobisClassifier(point, config.means, config.covariances)
      );

      newResults[datasetKey] = {
        bayesian: calculateError(bayesianPreds, labels),
        euclidean: calculateError(euclideanPreds, labels),
        mahalanobis: calculateError(mahalanobisPreds, labels),
        predictions: {
          bayesian: bayesianPreds,
          euclidean: euclideanPreds,
          mahalanobis: mahalanobisPreds
        }
      };
    });

    setDatasets(newDatasets);
    setResults(newResults);
  }, []);

  // Preparar datos para visualización
  const getScatterData = (datasetKey) => {
    if (!datasets[datasetKey]) return [];
    
    const { data, labels } = datasets[datasetKey];
    const colors = ['#8884d8', '#82ca9d', '#ffc658'];
    
    return data.map((point, idx) => ({
      x: point[0],
      y: point[1],
      class: labels[idx],
      fill: colors[labels[idx]]
    }));
  };

  const getResultsData = () => {
    return Object.keys(results).map(dataset => ({
      dataset,
      Bayesiano: parseFloat(results[dataset]?.bayesian || 0),
      Euclidiano: parseFloat(results[dataset]?.euclidean || 0),
      Mahalanobis: parseFloat(results[dataset]?.mahalanobis || 0)
    }));
  };

  return (
    <div className="p-8 max-w-7xl mx-auto">
      <h1 className="text-3xl font-bold mb-6 text-center">
        Ejercicio 7: Comparación de Clasificadores
      </h1>
      
      <div className="mb-8">
        <h2 className="text-2xl font-semibold mb-4">Descripción</h2>
        <p className="text-gray-700 mb-4">
          Este ejercicio compara tres tipos de clasificadores:
        </p>
        <ul className="list-disc pl-6 space-y-2 text-gray-700">
          <li><strong>Bayesiano:</strong> Usa la función discriminante gaussiana completa</li>
          <li><strong>Euclidiano:</strong> Clasifica según la distancia euclidiana más cercana</li>
          <li><strong>Mahalanobis:</strong> Usa la distancia de Mahalanobis</li>
        </ul>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        {/* Visualización de datos */}
        <div className="bg-white p-6 rounded-lg shadow-lg">
          <h3 className="text-xl font-semibold mb-4">
            Visualización de Datos
          </h3>
          
          <div className="mb-4">
            <label className="block text-sm font-medium mb-2">
              Seleccionar Dataset:
            </label>
            <select 
              value={currentDataset}
              onChange={(e) => setCurrentDataset(e.target.value)}
              className="w-full p-2 border rounded"
            >
              {Object.keys(datasetConfigs).map(key => (
                <option key={key} value={key}>
                  {datasetConfigs[key].name}
                </option>
              ))}
            </select>
          </div>

          <ResponsiveContainer width="100%" height={400}>
            <ScatterChart>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="x" type="number" domain={['dataMin - 2', 'dataMax + 2']} />
              <YAxis dataKey="y" type="number" domain={['dataMin - 2', 'dataMax + 2']} />
              <Tooltip />
              <Scatter 
                name="Clase 1" 
                data={getScatterData(currentDataset).filter(d => d.class === 0)}
                fill="#8884d8" 
              />
              <Scatter 
                name="Clase 2" 
                data={getScatterData(currentDataset).filter(d => d.class === 1)}
                fill="#82ca9d" 
              />
              <Scatter 
                name="Clase 3" 
                data={getScatterData(currentDataset).filter(d => d.class === 2)}
                fill="#ffc658" 
              />
            </ScatterChart>
          </ResponsiveContainer>
        </div>

        {/* Resultados de errores */}
        <div className="bg-white p-6 rounded-lg shadow-lg">
          <h3 className="text-xl font-semibold mb-4">
            Errores de Clasificación (%)
          </h3>
          
          <ResponsiveContainer width="100%" height={400}>
            <LineChart data={getResultsData()}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="dataset" />
              <YAxis />
              <Tooltip />
              <Legend />
              <Line type="monotone" dataKey="Bayesiano" stroke="#8884d8" strokeWidth={2} />
              <Line type="monotone" dataKey="Euclidiano" stroke="#82ca9d" strokeWidth={2} />
              <Line type="monotone" dataKey="Mahalanobis" stroke="#ffc658" strokeWidth={2} />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Tabla de resultados */}
      <div className="mt-8 bg-white p-6 rounded-lg shadow-lg">
        <h3 className="text-xl font-semibold mb-4">Tabla de Resultados</h3>
        <div className="overflow-x-auto">
          <table className="w-full border-collapse border border-gray-300">
            <thead>
              <tr className="bg-gray-100">
                <th className="border border-gray-300 px-4 py-2">Dataset</th>
                <th className="border border-gray-300 px-4 py-2">Descripción</th>
                <th className="border border-gray-300 px-4 py-2">Bayesiano (%)</th>
                <th className="border border-gray-300 px-4 py-2">Euclidiano (%)</th>
                <th className="border border-gray-300 px-4 py-2">Mahalanobis (%)</th>
              </tr>
            </thead>
            <tbody>
              {Object.keys(datasetConfigs).map(key => (
                <tr key={key}>
                  <td className="border border-gray-300 px-4 py-2 font-medium">{key}</td>
                  <td className="border border-gray-300 px-4 py-2">{datasetConfigs[key].name}</td>
                  <td className="border border-gray-300 px-4 py-2 text-center">
                    {results[key]?.bayesian || 'Calculando...'}
                  </td>
                  <td className="border border-gray-300 px-4 py-2 text-center">
                    {results[key]?.euclidean || 'Calculando...'}
                  </td>
                  <td className="border border-gray-300 px-4 py-2 text-center">
                    {results[key]?.mahalanobis || 'Calculando...'}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Análisis y conclusiones */}
      <div className="mt-8 bg-white p-6 rounded-lg shadow-lg">
        <h3 className="text-xl font-semibold mb-4">Análisis y Conclusiones</h3>
        <div className="space-y-4 text-gray-700">
          <p>
            <strong>1. Clasificador Bayesiano:</strong> Generalmente obtiene el mejor rendimiento 
            ya que utiliza toda la información estadística disponible (medias, covarianzas y probabilidades a priori).
          </p>
          <p>
            <strong>2. Clasificador Euclidiano:</strong> Su rendimiento depende de qué tan separadas 
            estén las clases y si las covarianzas son similares a la identidad.
          </p>
          <p>
            <strong>3. Clasificador Mahalanobis:</strong> Mejora sobre el euclidiano cuando las 
            covarianzas no son esféricas, pero puede ser inferior al Bayesiano que también 
            considera las probabilidades a priori.
          </p>
          <p>
            <strong>4. Efecto de la correlación:</strong> Cuando las variables están correlacionadas 
            (D2, D4), los clasificadores que ignoran esta información (Euclidiano) tienen peor rendimiento.
          </p>
          <p>
            <strong>5. Separación de clases:</strong> Cuando las clases están muy cerca (D3, D4), 
            todos los clasificadores sufren, pero el Bayesiano mantiene ventaja.
          </p>
        </div>
      </div>
    </div>
  );
};

export default PatternRecognitionExercise;