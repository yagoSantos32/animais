import { useState, useEffect } from 'react'

type EstadoAnimal = {
  titulo: string
  categoria: string
  urgente: boolean
}

function obterEstadoAnimal(
  animalEmAbandono: boolean,
  animalEmRisco: boolean
): EstadoAnimal {
  if (animalEmAbandono && animalEmRisco) {
    return {
      titulo: 'Animal abandonado em área de risco',
      categoria: 'Abandono e risco detectados',
      urgente: true,
    }
  }
  if (animalEmAbandono) {
    return {
      titulo: 'Animal potencialmente abandonado',
      categoria: 'Possível abandono',
      urgente: false,
    }
  }
  if (animalEmRisco) {
    return {
      titulo: 'Animal em situação de risco',
      categoria: 'Risco detectado',
      urgente: true,
    }
  }
  return {
    titulo: 'Animal detectado',
    categoria: 'Presença detectada',
    urgente: false,
  }
}

type AlertaAnimalProps = {
  animalEmAbandono: boolean
  animalEmRisco: boolean
  tipoAnimal?: string
  imagem?: string
  timestamp: string
  onAcionarResgate: () => void
  onAlarmeFalso: () => void
}

function AlertaAnimal({
  animalEmAbandono,
  animalEmRisco,
  tipoAnimal = 'Animal não identificado',
  imagem = "https://images.unsplash.com/photo-1548199973-03cce0bbc87b?w=900&h=506&fit=crop&auto=format",
  timestamp,
  onAcionarResgate,
  onAlarmeFalso,
}: AlertaAnimalProps) {
  const estado = obterEstadoAnimal(animalEmAbandono, animalEmRisco)

  return (
    <main className="min-h-screen bg-black flex items-center justify-center p-4 sm:p-8">
      <section className="w-full max-w-lg space-y-6">
        <header className="flex items-center gap-2">
          <span className="relative flex h-2 w-2" aria-hidden="true">
            <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-red-600 opacity-75" />
            <span className="relative inline-flex h-2 w-2 rounded-full bg-red-600" />
          </span>
          <span className="text-[11px] tracking-[0.25em] text-white/40 uppercase">
            Alerta ativo
          </span>
        </header>

        <figure className="relative w-full aspect-video overflow-hidden bg-zinc-900">
          <img
            src={imagem}
            alt={`${tipoAnimal} detectado pela câmera`}
            className="h-full w-full object-cover grayscale opacity-60"
          />
          <div
            className="absolute border border-red-600"
            style={{ top: '26%', left: '37%', width: '24%', height: '44%' }}
          />
          <figcaption className="absolute bottom-3 right-3 bg-black/50 px-2 py-1 font-mono text-[10px] tracking-widest text-white/60">
            {timestamp}
          </figcaption>
        </figure>

        <div className="space-y-2">
          <p className="text-[11px] tracking-[0.2em] text-red-500 uppercase">
            {tipoAnimal} - {estado.categoria}
          </p>
          <h1 className="text-3xl font-bold leading-tight tracking-tight text-white">
            {estado.titulo}
          </h1>
          {estado.urgente && (
            <p className="text-sm text-red-400">
              Esta ocorrência requer atendimento prioritário.
            </p>
          )}
        </div>

        <div className="flex flex-col gap-3 pt-2 sm:flex-row">
          <button onClick={onAcionarResgate} className="flex-1 bg-red-600 px-4 py-3 text-xs font-semibold tracking-[0.2em] text-white uppercase hover:bg-red-500">
            Acionar resgate
          </button>
          <button onClick={onAlarmeFalso} className="flex-1 border border-white/20 px-4 py-3 text-xs font-semibold tracking-[0.2em] text-white uppercase hover:border-white/40 hover:bg-white/10">
            Alarme falso
          </button>
        </div>
      </section>
    </main>
  )
}

export default function App() {
  const [ts, setTs] = useState('')
  const [animalDetectado, setAnimalDetectado] = useState(true)
  const [animalEmRisco, setAnimalEmRisco] = useState(true)
  const [animalEmAbandono, setAnimalEmAbandono] = useState(false)

  useEffect(() => {
    setTs(new Date().toLocaleTimeString('pt-BR'))
    const id = setInterval(() => setTs(new Date().toLocaleTimeString('pt-BR')), 1000)
    return () => clearInterval(id)
  }, [])

  if (!animalDetectado) {
    return (
      <div className="min-h-screen bg-black flex items-center justify-center p-8">
        <p className="text-3xl font-bold text-white">Nenhum animal detectado</p>
      </div>
    )
  }

  return (
    <AlertaAnimal
      animalEmAbandono={animalEmAbandono}
      animalEmRisco={animalEmRisco}
      tipoAnimal="Cachorro"
      timestamp={ts}
      onAcionarResgate={() => console.log('Resgate acionado')}
      onAlarmeFalso={() => setAnimalDetectado(false)}
    />
  )
}
