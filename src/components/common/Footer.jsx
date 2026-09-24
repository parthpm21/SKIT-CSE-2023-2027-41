import { Link } from "react-router-dom";
import { ShieldCheck } from "lucide-react";

function Footer() {
  return (
    <footer className="border-t border-gray-100 bg-white px-6 py-12">
      <div className="mx-auto flex max-w-7xl flex-col gap-8 md:flex-row md:items-center md:justify-between">

        {/* Brand */}
        <div>
          <Link
            to="/"
            className="group flex items-center gap-2.5"
          >
            <div className="flex h-9 w-9 items-center justify-center rounded-xl bg-purple-100 transition-transform duration-200 group-hover:scale-105">
              <ShieldCheck className="h-5 w-5 text-purple-600" />
            </div>

            <span className="text-lg font-bold text-gray-900">
              Truth<span className="text-purple-600">Lens</span>
            </span>
          </Link>

          <p className="mt-3 max-w-sm text-sm leading-6 text-gray-500">
            An explainable deep learning framework for digital media
            authenticity, manipulation detection, and localization.
          </p>
        </div>

        {/* Links */}
        <div className="flex flex-wrap gap-x-8 gap-y-3 text-sm">
          <Link
            to="/"
            className="text-gray-500 transition-colors duration-200 hover:text-purple-600"
          >
            Home
          </Link>

          <Link
            to="/analyze"
            className="text-gray-500 transition-colors duration-200 hover:text-purple-600"
          >
            Analyze
          </Link>

          <Link
            to="/about"
            className="text-gray-500 transition-colors duration-200 hover:text-purple-600"
          >
            About
          </Link>
        </div>

      </div>

      <div className="mx-auto mt-10 max-w-7xl border-t border-gray-100 pt-6">
        <p className="text-center text-xs text-gray-400">
          © {new Date().getFullYear()} TruthLens. Explainable digital media
          forensics.
        </p>
      </div>
    </footer>
  );
}

export default Footer;