"use client"

import { useEffect, useRef } from "react"
import { Shield } from "lucide-react"

export default function HeroAnimation() {
  const canvasRef = useRef<HTMLCanvasElement>(null)

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return

    const ctx = canvas.getContext("2d")
    if (!ctx) return

    // Set canvas dimensions
    const setCanvasDimensions = () => {
      canvas.width = canvas.clientWidth
      canvas.height = canvas.clientHeight
    }

    setCanvasDimensions()
    window.addEventListener("resize", setCanvasDimensions)

    // Node class for network nodes
    class Node {
      x: number
      y: number
      radius: number
      color: string
      vx: number
      vy: number
      type: "normal" | "malicious" | "secure"

      constructor(x: number, y: number, type: "normal" | "malicious" | "secure") {
        this.x = x
        this.y = y
        this.radius = type === "normal" ? 4 : 6
        this.color =
          type === "normal"
            ? "rgba(255, 255, 255, 0.7)"
            : type === "malicious"
              ? "rgba(239, 68, 68, 0.8)"
              : "rgba(16, 185, 129, 0.8)"
        this.vx = (Math.random() - 0.5) * 0.5
        this.vy = (Math.random() - 0.5) * 0.5
        this.type = type
      }

      update() {
        // Bounce off edges
        if (this.x + this.radius > canvas.width || this.x - this.radius < 0) {
          this.vx = -this.vx
        }

        if (this.y + this.radius > canvas.height || this.y - this.radius < 0) {
          this.vy = -this.vy
        }

        this.x += this.vx
        this.y += this.vy
      }

      draw() {
        if (!ctx) return

        ctx.beginPath()
        ctx.arc(this.x, this.y, this.radius, 0, Math.PI * 2)
        ctx.fillStyle = this.color
        ctx.fill()

        // Draw glow effect for special nodes
        if (this.type !== "normal") {
          ctx.beginPath()
          ctx.arc(this.x, this.y, this.radius * 2, 0, Math.PI * 2)
          const gradient = ctx.createRadialGradient(this.x, this.y, this.radius, this.x, this.y, this.radius * 2)
          gradient.addColorStop(0, this.type === "malicious" ? "rgba(239, 68, 68, 0.5)" : "rgba(16, 185, 129, 0.5)")
          gradient.addColorStop(1, "rgba(0, 0, 0, 0)")
          ctx.fillStyle = gradient
          ctx.fill()
        }
      }
    }

    // Connection class for lines between nodes
    class Connection {
      from: Node
      to: Node
      width: number
      color: string
      speed: number
      progress: number
      active: boolean

      constructor(from: Node, to: Node) {
        this.from = from
        this.to = to
        this.width = Math.random() * 1.5 + 0.5
        this.color =
          from.type === "malicious" || to.type === "malicious" ? "rgba(239, 68, 68, 0.5)" : "rgba(255, 255, 255, 0.2)"
        this.speed = Math.random() * 0.02 + 0.01
        this.progress = 0
        this.active = Math.random() > 0.5
      }

      update() {
        if (this.active) {
          this.progress += this.speed
          if (this.progress >= 1) {
            this.progress = 0
            this.active = Math.random() > 0.3
          }
        } else if (Math.random() > 0.99) {
          this.active = true
        }
      }

      draw() {
        if (!ctx) return

        const dx = this.to.x - this.from.x
        const dy = this.to.y - this.from.y
        const distance = Math.sqrt(dx * dx + dy * dy)

        if (distance > 150) return

        ctx.beginPath()
        ctx.moveTo(this.from.x, this.from.y)
        ctx.lineTo(this.to.x, this.to.y)
        ctx.strokeStyle = this.color
        ctx.lineWidth = this.width * (1 - distance / 150)
        ctx.stroke()

        // Draw data packet animation
        if (this.active) {
          const packetX = this.from.x + dx * this.progress
          const packetY = this.from.y + dy * this.progress

          ctx.beginPath()
          ctx.arc(packetX, packetY, 2, 0, Math.PI * 2)
          ctx.fillStyle =
            this.from.type === "malicious" || this.to.type === "malicious"
              ? "rgba(239, 68, 68, 0.8)"
              : "rgba(255, 255, 255, 0.8)"
          ctx.fill()
        }
      }
    }

    // Create nodes and connections
    const nodes: Node[] = []
    const connections: Connection[] = []

    const createNetwork = () => {
      // Clear existing
      nodes.length = 0
      connections.length = 0

      // Create nodes
      const nodeCount = Math.floor(canvas.width / 20)
      for (let i = 0; i < nodeCount; i++) {
        const type = Math.random() > 0.95 ? "malicious" : Math.random() > 0.9 ? "secure" : "normal"

        nodes.push(new Node(Math.random() * canvas.width, Math.random() * canvas.height, type))
      }

      // Create connections
      for (let i = 0; i < nodes.length; i++) {
        for (let j = i + 1; j < nodes.length; j++) {
          if (Math.random() > 0.85) {
            connections.push(new Connection(nodes[i], nodes[j]))
          }
        }
      }
    }

    createNetwork()
    window.addEventListener("resize", createNetwork)

    // Animation loop
    const animate = () => {
      if (!ctx) return

      // Clear canvas with semi-transparent background for trail effect
      ctx.fillStyle = "rgba(17, 24, 39, 0.2)"
      ctx.fillRect(0, 0, canvas.width, canvas.height)

      // Draw connections first (behind nodes)
      connections.forEach((connection) => {
        connection.update()
        connection.draw()
      })

      // Draw nodes
      nodes.forEach((node) => {
        node.update()
        node.draw()
      })

      // Draw shield in center
      const centerX = canvas.width / 2
      const centerY = canvas.height / 2

      ctx.beginPath()
      ctx.arc(centerX, centerY, 30, 0, Math.PI * 2)
      const gradient = ctx.createRadialGradient(centerX, centerY, 10, centerX, centerY, 40)
      gradient.addColorStop(0, "rgba(16, 185, 129, 0.2)")
      gradient.addColorStop(1, "rgba(16, 185, 129, 0)")
      ctx.fillStyle = gradient
      ctx.fill()

      // Draw shield icon
      ctx.save()
      ctx.translate(centerX - 12, centerY - 12)
      ctx.scale(1.5, 1.5)
      ctx.strokeStyle = "rgba(16, 185, 129, 0.8)"
      ctx.lineWidth = 2

      // Simple shield path
      ctx.beginPath()
      ctx.moveTo(8, 3)
      ctx.lineTo(16, 3)
      ctx.lineTo(16, 8)
      ctx.quadraticCurveTo(16, 16, 8, 21)
      ctx.quadraticCurveTo(0, 16, 0, 8)
      ctx.lineTo(0, 3)
      ctx.closePath()
      ctx.stroke()
      ctx.restore()

      requestAnimationFrame(animate)
    }

    animate()

    return () => {
      window.removeEventListener("resize", setCanvasDimensions)
      window.removeEventListener("resize", createNetwork)
    }
  }, [])

  return (
    <div className="relative w-full h-[400px] md:h-[500px]">
      <canvas ref={canvasRef} className="w-full h-full rounded-lg" />
      <div className="absolute inset-0 flex items-center justify-center pointer-events-none">
        <div className="flex flex-col items-center">
          <Shield className="h-16 w-16 text-emerald-500 opacity-70" />
        </div>
      </div>
    </div>
  )
}
