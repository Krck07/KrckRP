import { GameStateProvider } from './GameStateContext';
import ChatInterface from './ChatInterface';
import Sidebar from './Sidebar';
import './App.css';

function App() {
  return (
    <GameStateProvider>
      <div className="flex h-screen w-full bg-slate-950 text-slate-100 font-sans">
        {/* Main Chat Area */}
        <main className="flex-1 flex flex-col h-full relative">
          <header className="p-4 border-b border-slate-800 bg-slate-900 shadow-sm z-10 flex-shrink-0">
            <h1 className="text-xl font-bold tracking-widest text-slate-200 uppercase">
              KrckRP Engine
            </h1>
          </header>
          <div className="flex-1 overflow-hidden">
            <ChatInterface />
          </div>
        </main>

        {/* RPG Status Sidebar */}
        <aside className="w-80 lg:w-96 h-full border-l border-slate-800 bg-slate-900 overflow-y-auto flex-shrink-0 shadow-xl">
          <Sidebar />
        </aside>
      </div>
    </GameStateProvider>
  );
}

export default App;