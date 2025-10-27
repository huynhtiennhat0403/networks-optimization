import { useNavigate } from 'react-router-dom';
import { Zap, ListChecks, ArrowRight, CheckCircle2, Network } from 'lucide-react';

function Home() {
  const navigate = useNavigate();
  
  const features = [
    { icon: CheckCircle2, text: 'AI-Powered Predictions', color: 'text-primary-600' },
    { icon: CheckCircle2, text: 'Real-time Network Analysis', color: 'text-primary-600' },
    { icon: CheckCircle2, text: 'Multiple Prediction Modes', color: 'text-primary-600' },
    { icon: CheckCircle2, text: 'Vietnamese Context Scenarios', color: 'text-primary-600' },
  ];
  
  return (
    <div className="container mx-auto px-6 py-12">
      {/* Hero Section */}
      <div className="text-center mb-16 animate-fade-in">
        <div className="flex justify-center mb-6">
          <div className="p-4 bg-primary-100 rounded-full">
            <Network className="w-16 h-16 text-primary-600" />
          </div>
        </div>
        
        <h1 className="text-5xl font-bold text-gray-900 mb-4">
          Network Quality Predictor
        </h1>
        <p className="text-xl text-gray-600 mb-8 max-w-2xl mx-auto">
          Predict your network quality using AI-powered analysis. 
          Choose from smart input or pre-defined scenarios for instant results.
        </p>
        
        <div className="flex flex-wrap justify-center gap-4 mb-8">
          {features.map((feature, index) => (
            <div key={index} className="flex items-center space-x-2 bg-white px-4 py-2 rounded-lg shadow-sm border border-gray-200">
              <feature.icon className={`w-5 h-5 ${feature.color}`} />
              <span className="text-sm font-medium text-gray-700">{feature.text}</span>
            </div>
          ))}
        </div>
      </div>
      
      {/* Mode Selection Cards */}
      <div className="grid md:grid-cols-2 gap-8 max-w-5xl mx-auto">
        {/* Mode 1: Smart Input */}
        <div className="card hover:shadow-xl transition-all duration-300 animate-slide-up">
          <div className="flex items-start space-x-4 mb-4">
            <div className="p-3 bg-primary-100 rounded-lg">
              <Zap className="w-8 h-8 text-primary-600" />
            </div>
            <div>
              <h2 className="text-2xl font-bold text-gray-900 mb-2">Smart Input</h2>
              <p className="text-gray-600">
                Enter 5 basic network metrics. Our AI will estimate the remaining parameters 
                and provide accurate predictions.
              </p>
            </div>
          </div>
          
          <div className="space-y-2 mb-6">
            <div className="flex items-center text-sm text-gray-600">
              <CheckCircle2 className="w-4 h-4 text-success-500 mr-2" />
              <span>Quick and easy input</span>
            </div>
            <div className="flex items-center text-sm text-gray-600">
              <CheckCircle2 className="w-4 h-4 text-success-500 mr-2" />
              <span>AI-powered parameter estimation</span>
            </div>
            <div className="flex items-center text-sm text-gray-600">
              <CheckCircle2 className="w-4 h-4 text-success-500 mr-2" />
              <span>Context-aware predictions</span>
            </div>
          </div>
          
          <button
            onClick={() => navigate('/smart-input')}
            className="btn-primary w-full flex items-center justify-center space-x-2"
          >
            <span>Start Smart Input</span>
            <ArrowRight className="w-5 h-5" />
          </button>
        </div>
        
        {/* Mode 2: Scenarios */}
        <div className="card hover:shadow-xl transition-all duration-300 animate-slide-up" style={{ animationDelay: '0.1s' }}>
          <div className="flex items-start space-x-4 mb-4">
            <div className="p-3 bg-success-100 rounded-lg">
              <ListChecks className="w-8 h-8 text-success-600" />
            </div>
            <div>
              <h2 className="text-2xl font-bold text-gray-900 mb-2">Scenario Selection</h2>
              <p className="text-gray-600">
                Choose from 5 realistic Vietnamese network scenarios. 
                Perfect for testing and demos.
              </p>
            </div>
          </div>
          
          <div className="space-y-2 mb-6">
            <div className="flex items-center text-sm text-gray-600">
              <CheckCircle2 className="w-4 h-4 text-success-500 mr-2" />
              <span>5 pre-defined scenarios</span>
            </div>
            <div className="flex items-center text-sm text-gray-600">
              <CheckCircle2 className="w-4 h-4 text-success-500 mr-2" />
              <span>Vietnamese context (HCMC, highways, etc.)</span>
            </div>
            <div className="flex items-center text-sm text-gray-600">
              <CheckCircle2 className="w-4 h-4 text-success-500 mr-2" />
              <span>Instant predictions</span>
            </div>
          </div>
          
          <button
            onClick={() => navigate('/scenarios')}
            className="btn-primary w-full flex items-center justify-center space-x-2"
          >
            <span>Browse Scenarios</span>
            <ArrowRight className="w-5 h-5" />
          </button>
        </div>
      </div>
      
      {/* Info Section */}
      <div className="mt-16 text-center">
        <div className="max-w-3xl mx-auto bg-primary-50 border border-primary-200 rounded-xl p-8">
          <h3 className="text-2xl font-bold text-gray-900 mb-4">How It Works</h3>
          <p className="text-gray-700 mb-6">
            Our system uses a trained Random Forest model to predict network quality (Poor, Moderate, Good) 
            based on 16 network parameters. You can either provide basic metrics and let AI estimate the rest, 
            or select from realistic scenarios.
          </p>
          <div className="flex justify-center gap-4">
            <div className="bg-white px-6 py-3 rounded-lg shadow-sm border border-primary-200">
              <div className="text-3xl font-bold text-primary-600">95%</div>
              <div className="text-sm text-gray-600">Model Accuracy</div>
            </div>
            <div className="bg-white px-6 py-3 rounded-lg shadow-sm border border-primary-200">
              <div className="text-3xl font-bold text-primary-600">5</div>
              <div className="text-sm text-gray-600">Input Metrics</div>
            </div>
            <div className="bg-white px-6 py-3 rounded-lg shadow-sm border border-primary-200">
              <div className="text-3xl font-bold text-primary-600">16</div>
              <div className="text-sm text-gray-600">Total Parameters</div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

export default Home;