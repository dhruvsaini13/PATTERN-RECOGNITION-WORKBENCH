import createPlotlyComponent from 'react-plotly.js/factory'
import Plotly from 'plotly.js-dist-min'

// One shared Plotly component so the rest of the app never touches Plotly directly.
const Plot = createPlotlyComponent(Plotly)

export default function Chart({ data, layout = {}, height = 360 }) {
  return (
    <Plot
      data={data}
      layout={{
        autosize: true,
        height,
        margin: { l: 55, r: 20, t: 45, b: 55 },
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor: 'rgba(0,0,0,0)',
        ...layout,
      }}
      config={{ responsive: true, displaylogo: false }}
      useResizeHandler
      style={{ width: '100%' }}
    />
  )
}
