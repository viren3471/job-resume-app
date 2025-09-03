import axios from 'axios';

const api = axios.create({
  // This line is the fix.
  // In production (on Vercel), it uses a relative path "".
  // In development (on your PC), it uses "http://localhost:8000".
  baseURL: import.meta.env.PROD ? '' : 'http://localhost:8000'
});

export default api;