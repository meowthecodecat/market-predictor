import axios from 'axios'
const api = axios.create({ baseURL: '' }) // URL relative
export const getSummary = () => api.get('/api/summary').then(r=>r.data)
export const postRun = () => api.post('/api/run').then(r=>r.data)
export const getRunStatus = id => api.get(`/api/run/status/${id}`).then(r=>r.data)
export const getRunLogs = (id,n=200)=>api.get(`/api/run/logs/${id}`,{params:{last_n:n}}).then(r=>r.data)
export const getModels = () => api.get('/api/models').then(r=>r.data)
export default api
