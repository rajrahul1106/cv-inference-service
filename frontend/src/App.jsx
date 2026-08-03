import Dashboard from "./pages/Dashboard.jsx";

export default function App() {
  return (
    <div className="min-h-screen bg-slate-100 text-slate-800">
      <header className="bg-slate-900 text-white shadow">
        <div className="mx-auto max-w-7xl px-4 py-4 flex items-center gap-3">
          <span className="text-2xl" aria-hidden>
            🧠
          </span>
          <div>
            <h1 className="text-lg font-semibold leading-tight">
              CV Inference Dashboard
            </h1>
            <p className="text-xs text-slate-300">
              Submit an image URL and watch it run through YOLO / face / fire
              detection in real time.
            </p>
          </div>
        </div>
      </header>
      <main className="mx-auto max-w-7xl px-4 py-6">
        <Dashboard />
      </main>
    </div>
  );
}
