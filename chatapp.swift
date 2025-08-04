// Copyright © 2024 Apple Inc.

import MLX
import MLXLLM
import MLXLMCommon
import MLXRandom
import AVFoundation
import MediaPlayer
import SwiftUI

// MARK: - Models
struct Message: Identifiable, Codable, Equatable {
    var id: UUID
    let role: Role
    let content: String
    
    init(id: UUID = UUID(), role: Role, content: String) {
        self.id = id
        self.role = role
        self.content = content
    }
    
    enum Role: String, Codable { case user, assistant }
}

// MARK: - ModelConfiguration
struct ModelConfiguration {
    let id: String
    let overrideTokenizer: String? = nil
}

// MARK: - LLMEvaluator
@Observable
class LLMEvaluator {
    var running = false
    var output = ""
    var modelInfo = ""
    let modelConfiguration: ModelConfiguration = .init(id: "nidum/Nidum-Llama-3.2-3B-Uncensored-MLX-4bit")
    let generateParameters = GenerateParameters(temperature: 0.7, topP: 0.9, repetitionPenalty: 1.1)
    let maxTokens = 512
    
    enum LoadState { case idle, loaded(ModelContainer) }
    var loadState = LoadState.idle
    
    func load() async throws -> ModelContainer {
        if case .loaded(let container) = loadState { return container }
        
        do {
            MLX.GPU.set(cacheLimit: 1024 * 1024 * 1024)
            let container = try await LLMModelFactory.shared.loadContainer(
                configuration: .init(id: modelConfiguration.id, overrideTokenizer: modelConfiguration.overrideTokenizer)
            ) { [modelConfiguration] progress in
                Task { @MainActor in
                    self.modelInfo = "Downloading \(modelConfiguration.id): \(Int(progress.fractionCompleted * 100))%"
                    print(self.modelInfo)
                }
            }
            self.modelInfo = "Loaded \(modelConfiguration.id)"
            loadState = .loaded(container)
            return container
        } catch {
            print("Model loading failed: \(error)")
            throw error
        }
    }
    
    func generate(prompt: String) async -> String {
        guard !running else { return "Busy processing previous request" }
        running = true
        
        do {
            let container = try await load()
            let finalResponse = try await container.perform { context in
                let input = try await context.processor.prepare(input: .init(prompt: prompt))
                return try MLXLMCommon.generate(input: input, parameters: generateParameters, context: context) { tokens in
                    let text = context.tokenizer.decode(tokens: tokens)
                    Task { @MainActor in self.output = text }
                    return tokens.count >= maxTokens ? .stop : .more
                }
            }
            running = false
            return finalResponse.output.trimmingCharacters(in: .whitespacesAndNewlines)
        } catch {
            running = false
            print("Generation failed: \(error)")
            return "Error: \(error.localizedDescription)"
        }
    }
    
    deinit { MLX.GPU.set(cacheLimit: 0) }
}

// MARK: - Audio Manager
@MainActor
class AudioManager: NSObject, ObservableObject, AVSpeechSynthesizerDelegate {
    private let synthesizer = AVSpeechSynthesizer()
    private var continuation: CheckedContinuation<Void, Never>?
    @Published var isPlaying = false
    
    override init() {
        super.init()
        synthesizer.delegate = self
        setupAudioSession()
        setupRemoteCommands()
    }
    
    func speak(_ text: String) async {
        guard !text.isEmpty else { return }
        await withCheckedContinuation { continuation in
            self.continuation = continuation
            let utterance = AVSpeechUtterance(string: text)
            utterance.voice = AVSpeechSynthesisVoice(language: "en-US")
            utterance.rate = 0.5
            utterance.pitchMultiplier = 1.0
            utterance.volume = 1.0
            utterance.postUtteranceDelay = 0.2
            
            isPlaying = true
            synthesizer.speak(utterance)
            updateNowPlayingInfo(text: text)
            print("Speaking: \(text)")
        }
    }
    
    func pause() { synthesizer.pauseSpeaking(at: .immediate); isPlaying = false }
    func resume() { synthesizer.continueSpeaking(); isPlaying = true }
    func stop() { synthesizer.stopSpeaking(at: .immediate); isPlaying = false }
    
    private func setupAudioSession() {
        do {
            try AVAudioSession.sharedInstance().setCategory(.playback, mode: .spokenAudio, options: [.duckOthers, .allowAirPlay, .allowBluetooth])
            try AVAudioSession.sharedInstance().setActive(true, options: .notifyOthersOnDeactivation)
        } catch {
            print("Audio session setup failed: \(error)")
        }
        
        NotificationCenter.default.addObserver(self, selector: #selector(handleInterruption), name: AVAudioSession.interruptionNotification, object: nil)
    }
    
    @objc private func handleInterruption(_ notification: Notification) {
        guard let info = notification.userInfo,
              let typeValue = info[AVAudioSessionInterruptionTypeKey] as? UInt,
              let type = AVAudioSession.InterruptionType(rawValue: typeValue) else { return }
        
        switch type {
        case .began: pause()
        case .ended:
            if let optionsValue = info[AVAudioSessionInterruptionOptionKey] as? UInt,
               AVAudioSession.InterruptionOptions(rawValue: optionsValue).contains(.shouldResume) {
                resume()
            }
        @unknown default: break
        }
    }
    
    private func setupRemoteCommands() {
        let commandCenter = MPRemoteCommandCenter.shared()
        commandCenter.playCommand.addTarget { [unowned self] _ in isPlaying ? .commandFailed : (resume(), .success).1 }
        commandCenter.pauseCommand.addTarget { [unowned self] _ in !isPlaying ? .commandFailed : (pause(), .success).1 }
        commandCenter.togglePlayPauseCommand.addTarget { [unowned self] _ in isPlaying ? pause() : resume(); return .success }
        commandCenter.nextTrackCommand.addTarget { _ in .noActionableNowPlayingItem }
        UIApplication.shared.beginReceivingRemoteControlEvents()
    }
    
    private func updateNowPlayingInfo(text: String) {
        var info = [String: Any]()
        info[MPMediaItemPropertyTitle] = "Voice Chat"
        info[MPMediaItemPropertyArtist] = "Assistant"
        info[MPNowPlayingInfoPropertyPlaybackRate] = isPlaying ? 1.0 : 0.0
        info[MPMediaItemPropertyPlaybackDuration] = Double(text.count) / 10
        info[MPNowPlayingInfoPropertyElapsedPlaybackTime] = 0.0
        MPNowPlayingInfoCenter.default().nowPlayingInfo = info
    }
    
    nonisolated func speechSynthesizer(_ synthesizer: AVSpeechSynthesizer, didFinish utterance: AVSpeechUtterance) {
        Task { @MainActor in
            self.isPlaying = false
            self.continuation?.resume()
            self.continuation = nil
            print("Speech finished")
        }
    }
    
    nonisolated func speechSynthesizer(_ synthesizer: AVSpeechSynthesizer, didCancel utterance: AVSpeechUtterance) {
        Task { @MainActor in
            self.isPlaying = false
            self.continuation?.resume()
            self.continuation = nil
            print("Speech cancelled")
        }
    }
}

// MARK: - View Model
@MainActor
class VoiceChatViewModel: ObservableObject {
    static let shared = VoiceChatViewModel()
    private let llm = LLMEvaluator()
    private let audioManager = AudioManager()
    @Published var messages: [Message] = []
    @Published var isGenerating = false
    
    private init() {
        Task {
            do {
                _ = try await llm.load()
            } catch {
                print("Initialization failed: \(error)")
            }
        }
    }
    
    func sendMessage(userInput: String) async {
        guard !userInput.isEmpty, !isGenerating else { return }
        isGenerating = true
        messages.append(Message(role: .user, content: userInput))
        
        let response = await generateResponse(userInput: userInput)
        messages.append(Message(role: .assistant, content: response))
        await audioManager.speak(response)
        
        isGenerating = false
    }
    
    private func generateResponse(userInput: String) async -> String {
        let history = messages.map { "\($0.role.rawValue): \($0.content)" }.joined(separator: "\n")
        let prompt = """
        You are a helpful assistant.
        \(history)
        user: \(userInput)
        assistant:
        """
        return await llm.generate(prompt: prompt)
    }
}

// MARK: - App Delegate
class AppDelegate: NSObject, UIApplicationDelegate {
    func application(_ application: UIApplication, didFinishLaunchingWithOptions launchOptions: [UIApplication.LaunchOptionsKey: Any]?) -> Bool {
        print("App launched at \(Date())")
        return true
    }
}

// MARK: - App
@main
struct VoiceChatApp: App {
    @UIApplicationDelegateAdaptor(AppDelegate.self) var appDelegate
    
    var body: some Scene {
        WindowGroup {
            ContentView()
        }
    }
}

// MARK: - View
struct ContentView: View {
    @StateObject private var vm = VoiceChatViewModel.shared
    @State private var userInput = ""
    
    var body: some View {
        VStack {
            ScrollView {
                LazyVStack {
                    ForEach(vm.messages) { message in
                        HStack {
                            if message.role == .user {
                                Spacer()
                                Text(message.content)
                                    .padding()
                                    .background(Color.blue)
                                    .foregroundColor(.white)
                                    .cornerRadius(10)
                            } else {
                                Text(message.content)
                                    .padding()
                                    .background(Color.green)
                                    .foregroundColor(.white)
                                    .cornerRadius(10)
                                Spacer()
                            }
                        }
                    }
                }
            }
            
            HStack {
                TextField("Type your message...", text: $userInput)
                    .textFieldStyle(.roundedBorder)
                
                Button("Send") {
                    Task {
                        await vm.sendMessage(userInput: userInput)
                        userInput = ""
                    }
                }
                .disabled(vm.isGenerating || userInput.isEmpty)
            }
            .padding()
        }
    }
}
