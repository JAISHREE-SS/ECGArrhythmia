import { useState } from "react";
import { NavLink } from "@/components/NavLink";
import { useLocation } from "react-router-dom";
import {
  LayoutDashboard,
  Upload,
  Activity,
  History,
  BarChart3,
  Info,
  Heart,
  ChevronLeft,
  ChevronRight,
} from "lucide-react";

const navItems = [
  { title: "Dashboard", url: "/", icon: LayoutDashboard },
  { title: "Upload ECG", url: "/upload", icon: Upload },
  { title: "Real-Time Monitor", url: "/monitor", icon: Activity },
  { title: "Prediction History", url: "/history", icon: History },
  { title: "Model Insights", url: "/insights", icon: BarChart3 },
  { title: "About Project", url: "/about", icon: Info },
];

export function AppSidebar() {
  const [collapsed, setCollapsed] = useState(false);
  const location = useLocation();

  return (
    <aside
      className={`flex flex-col border-r bg-card transition-all duration-300 ${
        collapsed ? "w-16" : "w-64"
      }`}
    >
      {/* Logo */}
      <div className="flex items-center gap-3 border-b px-4 py-5">
        <div className="flex h-9 w-9 shrink-0 items-center justify-center rounded-lg bg-primary">
          <Heart className="h-5 w-5 text-primary-foreground" />
        </div>
        {!collapsed && (
          <div className="overflow-hidden">
            <h1 className="truncate text-sm font-bold text-foreground">
              Arrhythmia
            </h1>
            <p className="truncate text-[10px] text-muted-foreground">
              AI Detection System
            </p>
          </div>
        )}
      </div>

      {/* Navigation */}
      <nav className="flex-1 space-y-1 px-2 py-4">
        {navItems.map((item) => {
          const isActive = location.pathname === item.url;
          return (
            <NavLink
              key={item.url}
              to={item.url}
              end
              className={`flex items-center gap-3 rounded-lg px-3 py-2.5 text-sm font-medium transition-all duration-200 ${
                isActive
                  ? ""
                  : "text-muted-foreground hover:bg-secondary hover:text-foreground"
              }`}
              activeClassName="bg-primary/10 text-primary glow-primary"
            >
              <item.icon className="h-4.5 w-4.5 shrink-0" />
              {!collapsed && <span>{item.title}</span>}
            </NavLink>
          );
        })}
      </nav>

      {/* Status Banner */}
      {!collapsed && (
        <div className="mx-3 mb-3 rounded-lg bg-success/10 px-3 py-2">
          <div className="flex items-center gap-2">
            <span className="status-pulse h-2 w-2 rounded-full bg-success" />
            <span className="text-xs font-medium text-success">
              Model Running
            </span>
          </div>
        </div>
      )}

      {/* Collapse Toggle */}
      <button
        onClick={() => setCollapsed(!collapsed)}
        className="flex items-center justify-center border-t py-3 text-muted-foreground transition-colors hover:text-foreground"
      >
        {collapsed ? (
          <ChevronRight className="h-4 w-4" />
        ) : (
          <ChevronLeft className="h-4 w-4" />
        )}
      </button>
    </aside>
  );
}
