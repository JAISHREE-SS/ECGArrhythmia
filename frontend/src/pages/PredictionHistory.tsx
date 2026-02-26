import { useState } from "react";
import { motion } from "framer-motion";
import { Search, Download, FileDown } from "lucide-react";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";

const historyData = [
  { id: "P-4821", date: "2026-02-12", prediction: "Normal Sinus Rhythm", confidence: 98, risk: "Low" },
  { id: "P-4820", date: "2026-02-12", prediction: "Atrial Fibrillation", confidence: 94, risk: "High" },
  { id: "P-4819", date: "2026-02-11", prediction: "Normal Sinus Rhythm", confidence: 96, risk: "Low" },
  { id: "P-4818", date: "2026-02-11", prediction: "Premature Ventricular Contraction", confidence: 89, risk: "Medium" },
  { id: "P-4817", date: "2026-02-10", prediction: "Normal Sinus Rhythm", confidence: 97, risk: "Low" },
  { id: "P-4816", date: "2026-02-10", prediction: "Atrial Flutter", confidence: 91, risk: "High" },
  { id: "P-4815", date: "2026-02-09", prediction: "Normal Sinus Rhythm", confidence: 99, risk: "Low" },
  { id: "P-4814", date: "2026-02-09", prediction: "Supraventricular Tachycardia", confidence: 86, risk: "Medium" },
];

const PredictionHistory = () => {
  const [search, setSearch] = useState("");

  const filtered = historyData.filter(
    (item) =>
      item.id.toLowerCase().includes(search.toLowerCase()) ||
      item.prediction.toLowerCase().includes(search.toLowerCase())
  );

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold text-foreground">Prediction History</h1>
        <p className="text-muted-foreground">View past ECG analysis results</p>
      </div>

      <div className="flex items-center gap-3">
        <div className="relative flex-1 max-w-sm">
          <Search className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
          <Input
            placeholder="Search by ID or prediction..."
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            className="pl-9"
          />
        </div>
        <Button variant="outline" className="gap-2">
          <FileDown className="h-4 w-4" />
          Export
        </Button>
      </div>

      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="card-medical overflow-hidden"
      >
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead>Patient ID</TableHead>
              <TableHead>Date</TableHead>
              <TableHead>Prediction</TableHead>
              <TableHead>Confidence</TableHead>
              <TableHead>Risk</TableHead>
              <TableHead></TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {filtered.map((item) => (
              <TableRow key={item.id}>
                <TableCell className="font-medium">{item.id}</TableCell>
                <TableCell className="text-muted-foreground">{item.date}</TableCell>
                <TableCell>{item.prediction}</TableCell>
                <TableCell>
                  <span className="font-medium">{item.confidence}%</span>
                </TableCell>
                <TableCell>
                  <span
                    className={`rounded-full px-2.5 py-0.5 text-xs font-medium ${
                      item.risk === "Low"
                        ? "bg-success/10 text-success"
                        : item.risk === "Medium"
                        ? "bg-warning/10 text-warning"
                        : "bg-destructive/10 text-destructive"
                    }`}
                  >
                    {item.risk}
                  </span>
                </TableCell>
                <TableCell>
                  <Button variant="ghost" size="sm" className="gap-1">
                    <Download className="h-3.5 w-3.5" />
                    Report
                  </Button>
                </TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </motion.div>
    </div>
  );
};

export default PredictionHistory;
