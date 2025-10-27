import axios from 'axios'

const api = axios.create({
  baseURL: 'http://localhost:8000',
  timeout: 20000,
})

export const getSummary   = () => api.get('/api/summary').then(r => r.data)
export const postRun      = () => api.post('/api/run').then(r => r.data)
export const getRunStatus = (jobId) => api.get(`/api/run/status/${jobId}`).then(r => r.data)
export const getRunLogs   = (jobId, lastN = 200) =>
  api.get(`/api/run/logs/${jobId}`, { params: { last_n: lastN }}).then(r => r.data)
export const getModels    = () => api.get('/api/models').then(r => r.data)

export default api
