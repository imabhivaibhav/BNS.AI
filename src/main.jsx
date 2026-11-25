import { createRoot } from 'react-dom/client';
import App from './App.jsx';
import './styles.css';

// No StrictMode on purpose: the one-time SSO code must be redeemed exactly once.
createRoot(document.getElementById('root')).render(<App />);
