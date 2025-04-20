export async function predictNetworkTraffic(data: any) {
  try {
    const response = await fetch("http://localhost:5000/predict", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(data),
    })

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`)
    }

    return await response.json()
  } catch (error) {
    console.error("Error predicting network traffic:", error)
    throw error
  }
}

export async function fetchPlot(filename: string) {
  try {
    const response = await fetch(`http://localhost:5000/plots/${filename}`)

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`)
    }

    return await response.blob()
  } catch (error) {
    console.error("Error fetching plot:", error)
    throw error
  }
}
