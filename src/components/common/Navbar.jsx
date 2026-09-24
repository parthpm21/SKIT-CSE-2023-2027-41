import { Link, useLocation } from "react-router-dom";
import { ShieldCheck, ArrowRight } from "lucide-react";

function Navbar() {
  const location = useLocation();

  const navItems = [
    { label: "Home", path: "/" },
    { label: "Analyze", path: "/analyze" },
    { label: "About", path: "/about" },
  ];

  return (
    <nav className="fixed top-0 z-50 w-full px-4 pt-4">
      <div className="mx-auto flex max-w-7xl items-center justify-between rounded-2xl border border-white/70 bg-white/85 px-5 py-3 shadow-sm backdrop-blur-xl transition-shadow duration-300 hover:shadow-md md:px-7">

        {/* Logo */}
        <Link
          to="/"
          className="group flex items-center gap-2.5"
        >
          <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-purple-100 transition-all duration-300 group-hover:bg-purple-200 group-hover:scale-105">
            <ShieldCheck className="h-5 w-5 text-purple-600" />
          </div>

          <span className="text-xl font-bold tracking-tight text-gray-900">
            Truth<span className="text-purple-600">Lens</span>
          </span>
        </Link>

        {/* Navigation */}
        <div className="hidden items-center gap-2 md:flex">
          {navItems.map((item) => {
            const isActive = location.pathname === item.path;

            return (
              <Link
                key={item.path}
                to={item.path}
                className={`rounded-lg px-3 py-2 text-sm font-medium transition-all duration-200 ${
                  isActive
                    ? "bg-purple-50 text-purple-600"
                    : "text-gray-600 hover:bg-gray-50 hover:text-purple-600"
                }`}
              >
                {item.label}
              </Link>
            );
          })}
        </div>

        {/* CTA */}
        <Link
          to="/analyze"
          className="group flex items-center gap-2 rounded-full bg-purple-600 px-5 py-2.5 text-sm font-semibold text-white shadow-sm transition-all duration-200 hover:-translate-y-0.5 hover:bg-purple-700 hover:shadow-md"
        >
          Analyze Media
          <ArrowRight className="h-4 w-4 transition-transform duration-200 group-hover:translate-x-1" />
        </Link>

      </div>
    </nav>
  );
}

export default Navbar;