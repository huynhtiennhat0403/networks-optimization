import { BrowserRouter as Router, Routes, Route, Link, useLocation } from 'react-router-dom';
import { Network, Zap, ListChecks } from 'lucide-react';
import Home from './pages/Home';
import SmartInput from './pages/SmartInput';
import ScenarioSelect from './pages/ScenarioSelect';

function Navbar() {
  const location = useLocation();
  
  const isActive = (path) => {
    return location.pathname === path;
  };
  
  return (
    <nav className="bg-white shadow-md border-b border-gray-200">
      <div className="container mx-auto px-6 py-4">
        <div className="flex items-center justify-between">
          <Link to="/" className="flex items-center space-x-3 group">
            <Network className="w-8 h-8 text-primary-600 group-hover:text-primary-700 transition-colors" />
            <div>
              <h1 className="text-xl font-bold text-gray-900">Network Quality Predictor</h1>
              <p className="text-xs text-gray-500">AI-Powered Network Analysis</p>
            </div>
          </Link>
          
          <div className="flex items-center space-x-2">
            <Link
              to="/smart-input"
              className={`flex items-center space-x-2 px-4 py-2 rounded-lg transition-all ${
                isActive('/smart-input')
                  ? 'bg-primary-600 text-white shadow-md'
                  : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
              }`}
            >
              <Zap className="w-4 h-4" />
              <span className="font-medium">Smart Input</span>
            </Link>
            
            <Link
              to="/scenarios"
              className={`flex items-center space-x-2 px-4 py-2 rounded-lg transition-all ${
                isActive('/scenarios')
                  ? 'bg-primary-600 text-white shadow-md'
                  : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
              }`}
            >
              <ListChecks className="w-4 h-4" />
              <span className="font-medium">Scenarios</span>
            </Link>
          </div>
        </div>
      </div>
    </nav>
  );
}

function Footer() {
  return (
    <footer className="bg-gray-50 border-t border-gray-200 mt-auto">
      <div className="container mx-auto px-6 py-6">
        <div className="flex items-center justify-between text-sm text-gray-600">
          <p>© 2024 Network Quality Predictor. Built with React + FastAPI.</p>
          <div className="flex items-center space-x-4">
            <a href="https://github.com" className="hover:text-primary-600 transition-colors">
              GitHub
            </a>
            <a href="/docs" className="hover:text-primary-600 transition-colors">
              API Docs
            </a>
          </div>
        </div>
      </div>
    </footer>
  );
}

function App() {
  return (
    <Router>
      <div className="min-h-screen flex flex-col">
        <Navbar />
        
        <main className="flex-1">
          <Routes>
            <Route path="/" element={<Home />} />
            <Route path="/smart-input" element={<SmartInput />} />
            <Route path="/scenarios" element={<ScenarioSelect />} />
          </Routes>
        </main>
        
        <Footer />
      </div>
    </Router>
  );
}

export default App;