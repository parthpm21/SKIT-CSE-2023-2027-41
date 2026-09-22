import { Link } from "react-router-dom";
import { ShieldCheck, ArrowRight } from "lucide-react";

function Navbar() {
  return (
    <nav className="fixed top-0 z-50 w-full px-4 pt-4">
      <div className="mx-auto flex max-w-7xl items-center justify-between rounded-2xl border border-white/70 bg-white/85 px-5 py-3 shadow-sm backdrop-blur-xl md:px-7">

        {/* Logo */}
        <Link to="/" className="flex items-center gap-2.5">
          <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-purple-100">
            <ShieldCheck className="h-5 w-5 text-purple-600" />
          </div>

          <div>
            <span className="text-xl font-bold tracking-tight text-gray-900">
              Truth<span className="text-purple-600">Lens</span>
            </span>
          </div>
        </Link>

        {/* Navigation */}
        <div className="hidden items-center gap-8 md:flex">
          <Link
            to="/"
            className="text-sm font-medium text-purple-600"
          >
            Home
          </Link>

          <Link
            to="/analyze"
            className="text-sm font-medium text-gray-600 transition hover:text-purple-600"
          >
            Analyze
          </Link>

          <Link
            to="/how-it-works"
            className="text-sm font-medium text-gray-600 transition hover:text-purple-600"
          >
            How It Works
          </Link>

          <Link
            to="/about"
            className="text-sm font-medium text-gray-600 transition hover:text-purple-600"
          >
            About
          </Link>
        </div>

        {/* CTA */}
        <Link
          to="/analyze"
          className="group flex items-center gap-2 rounded-full bg-purple-600 px-5 py-2.5 text-sm font-semibold text-white shadow-sm transition hover:bg-purple-700"
        >
          Analyze Media
          <ArrowRight className="h-4 w-4 transition-transform group-hover:translate-x-0.5" />
        </Link>

      </div>
    </nav>
  );
}

export default Navbar;