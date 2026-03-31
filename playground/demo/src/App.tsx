import { Hero } from './components/Hero'
import { ParadigmShift } from './components/ParadigmShift'
import { MarketSolutions } from './components/MarketSolutions'
import { OurSolution } from './components/OurSolution'
import { Architecture } from './components/Architecture'
import { CoreConcepts } from './components/CoreConcepts'
import { AgentHarness } from './components/AgentHarness'
import { Replay } from './components/Replay/Replay'
import { TopologyPlugins } from './components/TopologyPlugins'
import { Integration } from './components/Integration'
import { Roadmap } from './components/Roadmap'
import { SideNav } from './components/shared/SideNav'

function Divider() {
  return (
    <div className="max-w-6xl mx-auto px-6">
      <div className="h-px bg-gradient-to-r from-transparent via-gray-800 to-transparent" />
    </div>
  )
}

function App() {
  return (
    <div className="bg-gray-950 text-white min-h-screen antialiased">
      <SideNav />
      <Hero />
      <Divider />
      <ParadigmShift />
      <Divider />
      <MarketSolutions />
      <Divider />
      <OurSolution />
      <Divider />
      <Architecture />
      <Divider />
      <CoreConcepts />
      <Divider />
      <AgentHarness />
      <Divider />
      <TopologyPlugins />
      <Divider />
      <Replay />
      <Divider />
      <Integration />
      <Divider />
      <Roadmap />

      {/* Footer */}
      <footer className="py-12 px-6 text-center border-t border-gray-900">
        <p className="text-sm text-gray-600 font-mono">
          AG2 V2 Network Framework — Internal Preview
        </p>
      </footer>
    </div>
  )
}

export default App
