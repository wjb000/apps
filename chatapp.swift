// Copyright © 2024 Apple Inc.

import MLX
import MLXLLM
import MLXLMCommon
import MLXRandom
import AVFoundation
import SwiftUI

// MARK: - Models
struct Message: Identifiable, Codable, Equatable {
    var id: UUID = UUID()
    let role: Role
    let content: String
    
    enum Role: String, Codable { case user, assistant }
}

// MARK: - ModelConfiguration
struct ModelConfiguration {
    let id: String = "nidum/Nidum-Llama-3.2-3B-Uncensored-MLX-4bit"
}

// MARK: - LLMEvaluator
@Observable
class LLMEvaluator {
    var running = false
    let modelConfiguration = ModelConfiguration()
    let generateParameters = GenerateParameters(temperature: 0.7, topP: 0.9, repetitionPenalty: 1.1)
    let maxTokens = 512
    
    enum LoadState { case idle, loaded(ModelContainer) }
    var loadState = LoadState.idle
    
    func load() async throws -> ModelContainer {
        if case .loaded(let container) = loadState { return container }
        
        MLX.GPU.set(cacheLimit: 1024 * 1024 * 1024)
        let container = try await LLMModelFactory.shared.loadContainer(
            configuration: .init(id: modelConfiguration.id)
        ) { [self] progress in
            print("Downloading \(modelConfiguration.id): \(Int(progress.fractionCompleted * 100))%")
        }
        print("Loaded \(modelConfiguration.id)")
        loadState = .loaded(container)
        return container
    }
    
    func generate(prompt: String) async -> String {
        guard !running else { return "" }
        running = true
        
        do {
            let container = try await load()
            let finalResponse = try await container.perform { context in
                let input = try await context.processor.prepare(input: .init(prompt: prompt))
                return try MLXLMCommon.generate(input: input, parameters: generateParameters, context: context) { tokens in
                    let text = context.tokenizer.decode(tokens: tokens)
                    return tokens.count >= maxTokens ? .stop : .more
                }
            }
            running = false
            return finalResponse.output.trimmingCharacters(in: .whitespacesAndNewlines)
        } catch {
            running = false
            print("Error: \(error)")
            return ""
        }
    }
}

// MARK: - Audio Manager
@MainActor
class AudioManager: NSObject, ObservableObject {
    private let synthesizer = AVSpeechSynthesizer()
    @Published var isPlaying = false
    
    func speak(_ text: String) async {
        guard !text.isEmpty else { return }
        let utterance = AVSpeechUtterance(string: text)
        utterance.voice = AVSpeechSynthesisVoice(language: "en-US")
        utterance.rate = 0.5
        isPlaying = true
        synthesizer.speak(utterance)
        print("Speaking: \(text)")
        await waitForCompletion()
        isPlaying = false
    }
    
    private func waitForCompletion() async {
        await withCheckedContinuation { continuation in
            synthesizer.delegate = self
            self.continuation = continuation
        }
    }
    
    private var continuation: CheckedContinuation<Void, Never>?
}

extension AudioManager: AVSpeechSynthesizerDelegate {
    nonisolated func speechSynthesizer(_ synthesizer: AVSpeechSynthesizer, didFinish utterance: AVSpeechUtterance) {
        Task { @MainActor in continuation?.resume() }
    }
    
    nonisolated func speechSynthesizer(_ synthesizer: AVSpeechSynthesizer, didCancel utterance: AVSpeechUtterance) {
        Task { @MainActor in continuation?.resume() }
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
        Task { try? await llm.load() }
    }
    
    func sendMessage(userInput: String) async {
        guard !userInput.isEmpty, !isGenerating else { return }
        isGenerating = true
        messages.append(Message(role: .user, content: userInput))
        
        let response = await generateResponse(userInput: userInput)
        if !response.isEmpty {
            messages.append(Message(role: .assistant, content: response))
            await audioManager.speak(response)
        }
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

// MARK: - App
@main
struct VoiceChatApp: App {
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
                    let input = userInput
                    userInput = ""
                    Task {
                        await vm.sendMessage(userInput: input)
                    }
                }
                .disabled(vm.isGenerating || userInput.isEmpty)
            }
            .padding()
        }
    }
}
