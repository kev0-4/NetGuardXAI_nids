import Link from "next/link"
import { ArrowRight, Shield, Database, LineChart, Lock, Server, Zap, Github } from "lucide-react"
import { Button } from "@/components/ui/button"
import HeroAnimation from "@/components/hero-animation"
import FeatureCard from "@/components/feature-card"

export default function LandingPage() {
  return (
    <div className="flex flex-col min-h-screen bg-gradient-to-b from-gray-900 to-gray-950">
      <header className="container mx-auto py-6 px-4 flex justify-between items-center">
        <div className="flex items-center gap-2">
          <Shield className="h-8 w-8 text-emerald-500" />
          <span className="text-xl font-bold text-white">NetGuardXAI</span>
        </div>
        <nav className="hidden md:flex items-center gap-8">
          <Link href="/" className="text-gray-300 hover:text-white transition-colors">
            Home
          </Link>
          <Link href="#features" className="text-gray-300 hover:text-white transition-colors">
            Features
          </Link>
          <Link href="#why-us" className="text-gray-300 hover:text-white transition-colors">
            Why Us
          </Link>
          <Link href="#references" className="text-gray-300 hover:text-white transition-colors">
            References
          </Link>
        </nav>
        <div className="flex items-center gap-3">
          <a href="https://github.com" target="_blank" rel="noopener noreferrer">
            <Button
              variant="outline"
              size="icon"
              className="bg-transparent border-gray-700 text-gray-300 hover:text-white hover:bg-gray-800"
            >
              <Github className="h-5 w-5" />
              <span className="sr-only">GitHub Repository</span>
            </Button>
          </a>
          <Link href="/predict">
            <Button
              variant="outline"
              className="bg-transparent border-emerald-500 text-emerald-500 hover:bg-emerald-500/10"
            >
              Try Demo
            </Button>
          </Link>
        </div>
      </header>

      <main className="flex-1">
        {/* Hero Section */}
        <section className="container mx-auto py-20 px-4 flex flex-col lg:flex-row items-center gap-12">
          <div className="flex-1 space-y-6">
            <div className="inline-block px-3 py-1 rounded-full bg-emerald-500/10 text-emerald-500 text-sm font-medium mb-2">
              Powered by Explainable AI
            </div>
            <h1 className="text-4xl md:text-5xl lg:text-6xl font-bold text-white leading-tight">
              Network Intrusion Detection with <span className="text-emerald-500">Transparency</span>
            </h1>
            <p className="text-xl text-gray-300 max-w-2xl">
              Our advanced system not only detects network intrusions but explains its decisions, providing security
              analysts with actionable insights and trust in AI-driven security.
            </p>
            <div className="flex flex-col sm:flex-row gap-4 pt-4">
              <Link href="/predict">
                <Button size="lg" className="bg-emerald-500 hover:bg-emerald-600 text-white">
                  Try the Demo <ArrowRight className="ml-2 h-4 w-4" />
                </Button>
              </Link>
              <Link href="#features">
                <Button
                  size="lg"
                  variant="outline"
                  className="bg-transparent border-gray-600 text-white hover:bg-gray-800"
                >
                  Learn More
                </Button>
              </Link>
            </div>
          </div>
          <div className="flex-1 flex justify-center">
            <HeroAnimation />
          </div>
        </section>

        {/* Features Section */}
        <section id="features" className="py-20 bg-gray-900/50">
          <div className="container mx-auto px-4">
            <div className="text-center mb-16">
              <h2 className="text-3xl md:text-4xl font-bold text-white mb-4">Key Features</h2>
              <p className="text-xl text-gray-300 max-w-3xl mx-auto">
                Our Network Intrusion Detection System combines cutting-edge machine learning with explainable AI to
                provide transparent, accurate, and actionable security insights.
              </p>
            </div>

            <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-8">
              <FeatureCard
                icon={<Shield className="h-10 w-10 text-emerald-500" />}
                title="Real-time Detection"
                description="Identify potential network intrusions as they happen with our high-performance detection engine."
              />
              <FeatureCard
                icon={<LineChart className="h-10 w-10 text-purple-500" />}
                title="Explainable Results"
                description="Understand why our model made specific predictions with detailed feature importance visualizations."
              />
              <FeatureCard
                icon={<Database className="h-10 w-10 text-blue-500" />}
                title="Comprehensive Analysis"
                description="Analyze network traffic patterns with multiple XAI techniques including LIME and Integrated Gradients."
              />
              <FeatureCard
                icon={<Lock className="h-10 w-10 text-red-500" />}
                title="Enhanced Security"
                description="Improve your network security posture with AI-driven insights and recommendations."
              />
              <FeatureCard
                icon={<Server className="h-10 w-10 text-yellow-500" />}
                title="Scalable Architecture"
                description="Handle enterprise-level network traffic with our optimized and scalable detection system."
              />
              <FeatureCard
                icon={<Zap className="h-10 w-10 text-cyan-500" />}
                title="Low Latency"
                description="Get results fast with our optimized prediction pipeline, essential for time-sensitive security operations."
              />
            </div>
          </div>
        </section>

        {/* Why Our Model Stands Out */}
        <section id="why-us" className="py-20 bg-gradient-to-b from-gray-900 to-gray-950">
          <div className="container mx-auto px-4">
            <div className="text-center mb-16">
              <h2 className="text-3xl md:text-4xl font-bold text-white mb-4">Why Our Model Stands Out</h2>
              <p className="text-xl text-gray-300 max-w-3xl mx-auto">
                Traditional "black box" AI models leave security teams in the dark. Our approach brings transparency to
                network security.
              </p>
            </div>

            <div className="grid md:grid-cols-2 gap-12 items-center">
              <div className="space-y-6">
                <div className="bg-gray-800/50 p-6 rounded-xl border border-gray-700">
                  <h3 className="text-xl font-bold text-white mb-3">Transparency First</h3>
                  <p className="text-gray-300">
                    Our model doesn't just tell you what it found—it shows you why. Every prediction comes with detailed
                    explanations of which network features influenced the decision.
                  </p>
                </div>

                <div className="bg-gray-800/50 p-6 rounded-xl border border-gray-700">
                  <h3 className="text-xl font-bold text-white mb-3">Multiple XAI Techniques</h3>
                  <p className="text-gray-300">
                    We combine LIME and Integrated Gradients to provide complementary explanations, giving you a more
                    complete understanding of model decisions.
                  </p>
                </div>

                <div className="bg-gray-800/50 p-6 rounded-xl border border-gray-700">
                  <h3 className="text-xl font-bold text-white mb-3">Reduced False Positives</h3>
                  <p className="text-gray-300">
                    By understanding why our model makes predictions, we've optimized it to minimize false alarms while
                    maintaining high detection rates for actual threats.
                  </p>
                </div>
              </div>

              <div className="relative">
                <div className="absolute -inset-0.5 bg-gradient-to-r from-emerald-500 to-blue-500 rounded-lg blur opacity-30"></div>
                <div className="relative bg-gray-800 p-8 rounded-lg border border-gray-700">
                  <div className="space-y-6">
                    <div className="flex items-start gap-4">
                      <div className="bg-emerald-500/20 p-3 rounded-full">
                        <Shield className="h-6 w-6 text-emerald-500" />
                      </div>
                      <div>
                        <h4 className="text-lg font-semibold text-white">Human-AI Collaboration</h4>
                        <p className="text-gray-300">
                          Our system is designed to work with security analysts, not replace them.
                        </p>
                      </div>
                    </div>

                    <div className="flex items-start gap-4">
                      <div className="bg-blue-500/20 p-3 rounded-full">
                        <LineChart className="h-6 w-6 text-blue-500" />
                      </div>
                      <div>
                        <h4 className="text-lg font-semibold text-white">Visual Explanations</h4>
                        <p className="text-gray-300">
                          Intuitive visualizations make complex model decisions easy to understand.
                        </p>
                      </div>
                    </div>

                    <div className="flex items-start gap-4">
                      <div className="bg-purple-500/20 p-3 rounded-full">
                        <Database className="h-6 w-6 text-purple-500" />
                      </div>
                      <div>
                        <h4 className="text-lg font-semibold text-white">Continuous Learning</h4>
                        <p className="text-gray-300">
                          Our model improves over time based on feedback and new threat patterns.
                        </p>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </section>

        {/* References Section */}
        <section id="references" className="py-20 bg-gray-900/50">
          <div className="container mx-auto px-4">
            <div className="text-center mb-16">
              <h2 className="text-3xl md:text-4xl font-bold text-white mb-4">Research & References</h2>
              <p className="text-xl text-gray-300 max-w-3xl mx-auto">
                Our work builds on cutting-edge research in network security and explainable AI.
              </p>
            </div>

            <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-8">
              <div className="bg-gray-800/50 p-6 rounded-xl border border-gray-700 hover:border-emerald-500/50 transition-colors">
                <h3 className="text-xl font-bold text-white mb-3">
                  LIME: Local Interpretable Model-agnostic Explanations
                </h3>
                <p className="text-gray-300 mb-4">
                  Ribeiro, M. T., Singh, S., & Guestrin, C. (2016). "Why should I trust you?": Explaining the
                  predictions of any classifier.
                </p>
                <a
                  href="https://arxiv.org/abs/1602.04938"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-emerald-500 hover:text-emerald-400 inline-flex items-center"
                >
                  Read paper <ArrowRight className="ml-1 h-4 w-4" />
                </a>
              </div>

              <div className="bg-gray-800/50 p-6 rounded-xl border border-gray-700 hover:border-emerald-500/50 transition-colors">
                <h3 className="text-xl font-bold text-white mb-3">Integrated Gradients</h3>
                <p className="text-gray-300 mb-4">
                  Sundararajan, M., Taly, A., & Yan, Q. (2017). Axiomatic attribution for deep networks.
                </p>
                <a
                  href="https://arxiv.org/abs/1703.01365"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-emerald-500 hover:text-emerald-400 inline-flex items-center"
                >
                  Read paper <ArrowRight className="ml-1 h-4 w-4" />
                </a>
              </div>

              <div className="bg-gray-800/50 p-6 rounded-xl border border-gray-700 hover:border-emerald-500/50 transition-colors">
                <h3 className="text-xl font-bold text-white mb-3">Network Intrusion Detection with Deep Learning</h3>
                <p className="text-gray-300 mb-4">
                  Recent advances in applying deep learning techniques to improve network intrusion detection systems.
                </p>
                <a
                  href="https://ieeexplore.ieee.org/document/8171733"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-emerald-500 hover:text-emerald-400 inline-flex items-center"
                >
                  Read paper <ArrowRight className="ml-1 h-4 w-4" />
                </a>
              </div>

              <div className="bg-gray-800/50 p-6 rounded-xl border border-gray-700 hover:border-emerald-500/50 transition-colors">
                <h3 className="text-xl font-bold text-white mb-3">Explainable AI for Security Applications</h3>
                <p className="text-gray-300 mb-4">
                  A comprehensive survey of XAI techniques applied to cybersecurity and network intrusion detection.
                </p>
                <a href="#" className="text-emerald-500 hover:text-emerald-400 inline-flex items-center">
                  Read paper <ArrowRight className="ml-1 h-4 w-4" />
                </a>
              </div>

              <div className="bg-gray-800/50 p-6 rounded-xl border border-gray-700 hover:border-emerald-500/50 transition-colors">
                <h3 className="text-xl font-bold text-white mb-3">Feature Selection for Network Security</h3>
                <p className="text-gray-300 mb-4">
                  Optimizing feature selection for network intrusion detection systems to improve accuracy and
                  performance.
                </p>
                <a href="#" className="text-emerald-500 hover:text-emerald-400 inline-flex items-center">
                  Read paper <ArrowRight className="ml-1 h-4 w-4" />
                </a>
              </div>

              <div className="bg-gray-800/50 p-6 rounded-xl border border-gray-700 hover:border-emerald-500/50 transition-colors">
                <h3 className="text-xl font-bold text-white mb-3">Human-AI Collaboration in Cybersecurity</h3>
                <p className="text-gray-300 mb-4">
                  Research on effective collaboration between security analysts and AI systems for improved threat
                  detection.
                </p>
                <a href="#" className="text-emerald-500 hover:text-emerald-400 inline-flex items-center">
                  Read paper <ArrowRight className="ml-1 h-4 w-4" />
                </a>
              </div>
            </div>
          </div>
        </section>

        {/* CTA Section */}
        <section className="py-20 bg-gradient-to-b from-gray-900 to-gray-950">
          <div className="container mx-auto px-4 text-center">
            <h2 className="text-3xl md:text-4xl font-bold text-white mb-6">
              Ready to try our Network Intrusion Detection System?
            </h2>
            <p className="text-xl text-gray-300 max-w-3xl mx-auto mb-8">
              Experience the power of explainable AI in network security with our interactive demo.
            </p>
            <Link href="/predict">
              <Button size="lg" className="bg-emerald-500 hover:bg-emerald-600 text-white">
                Try the Demo <ArrowRight className="ml-2 h-4 w-4" />
              </Button>
            </Link>
          </div>
        </section>
      </main>

      <footer className="bg-gray-950 py-12 border-t border-gray-800">
        <div className="container mx-auto px-4">
          <div className="flex flex-col md:flex-row justify-between items-center">
            <div className="flex items-center gap-2 mb-6 md:mb-0">
              <Shield className="h-8 w-8 text-emerald-500" />
              <span className="text-xl font-bold text-white">NetGuardXAI</span>
            </div>
            <div className="flex flex-col md:flex-row gap-4 md:gap-8 items-center">
              <Link href="/" className="text-gray-400 hover:text-white transition-colors">
                Home
              </Link>
              <Link href="#features" className="text-gray-400 hover:text-white transition-colors">
                Features
              </Link>
              <Link href="#why-us" className="text-gray-400 hover:text-white transition-colors">
                Why Us
              </Link>
              <Link href="#references" className="text-gray-400 hover:text-white transition-colors">
                References
              </Link>
              <Link href="/predict" className="text-gray-400 hover:text-white transition-colors">
                Demo
              </Link>
            </div>
          </div>
          <div className="mt-8 pt-8 border-t border-gray-800 text-center text-gray-400">
            <p>© {new Date().getFullYear()} NetGuardXAI. All rights reserved.</p>
          </div>
        </div>
      </footer>
    </div>
  )
}
