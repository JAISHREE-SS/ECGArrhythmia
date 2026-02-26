import { motion } from "framer-motion";
import { Heart, Brain, Shield, Zap } from "lucide-react";

const About = () => {
  return (
    <div className="space-y-8">
      <motion.div initial={{ opacity: 0, y: -10 }} animate={{ opacity: 1, y: 0 }}>
        <h1 className="text-2xl font-bold text-foreground">About the Project</h1>
        <p className="text-muted-foreground">
          AI-Based Arrhythmia Detection System
        </p>
      </motion.div>

      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="card-medical gradient-medical p-8"
      >
        <div className="mx-auto max-w-2xl space-y-4 text-center">
          <Heart className="mx-auto h-12 w-12 text-primary" />
          <h2 className="text-xl font-bold text-foreground">
            Early Detection. Smarter Diagnosis.
          </h2>
          <p className="text-sm leading-relaxed text-muted-foreground">
            This system leverages state-of-the-art deep learning models to
            detect cardiac arrhythmias from ECG signals in real-time. Our goal is
            to assist healthcare professionals with fast, accurate, and
            accessible heart rhythm analysis.
          </p>
        </div>
      </motion.div>

      <div className="grid gap-6 sm:grid-cols-2">
        {[
          {
            icon: Brain,
            title: "Deep Learning Model",
            desc: "1D-CNN with residual connections trained on 100K+ ECG recordings for multi-class arrhythmia classification.",
          },
          {
            icon: Zap,
            title: "Real-Time Analysis",
            desc: "Process and classify ECG signals in under 200ms, enabling continuous monitoring and instant alerts.",
          },
          {
            icon: Shield,
            title: "Clinical-Grade Accuracy",
            desc: "97.3% accuracy with high precision and recall across all arrhythmia types, validated against expert annotations.",
          },
          {
            icon: Heart,
            title: "Multi-Class Detection",
            desc: "Identifies Normal Sinus Rhythm, Atrial Fibrillation, Atrial Flutter, PVC, SVT, and more.",
          },
        ].map((item, i) => (
          <motion.div
            key={item.title}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.1 * i }}
            className="card-medical p-6"
          >
            <item.icon className="mb-3 h-8 w-8 text-primary" />
            <h3 className="mb-2 font-semibold text-foreground">{item.title}</h3>
            <p className="text-sm text-muted-foreground">{item.desc}</p>
          </motion.div>
        ))}
      </div>
    </div>
  );
};

export default About;
