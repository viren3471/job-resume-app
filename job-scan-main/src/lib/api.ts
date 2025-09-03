import axios from "axios";

const api = axios.create({
  baseURL: "http://localhost:8000", // backend URL
});

export default api;
