import SwiftUI
import SafariServices

/// Идентификатор расширения: <bundle id приложения>.Blocker
let blockerIdentifier = (Bundle.main.bundleIdentifier ?? "com.example.zeroads") + ".Blocker"

struct ContentView: View {
    @State private var isEnabled: Bool?
    @State private var message = ""

    var body: some View {
        NavigationStack {
            List {
                Section("Safari") {
                    HStack {
                        Image(systemName: isEnabled == true ? "checkmark.shield.fill" : "shield.slash")
                            .foregroundStyle(isEnabled == true ? .green : .orange)
                            .font(.title)
                        VStack(alignment: .leading) {
                            Text(isEnabled == true ? "Блокировка включена" : "Блокировка выключена")
                                .font(.headline)
                            if isEnabled != true {
                                Text("Настройки → Приложения → Safari → Расширения → ZeroAds → включить")
                                    .font(.footnote).foregroundStyle(.secondary)
                            }
                        }
                    }
                    Button("Обновить правила", action: reload)
                    if !message.isEmpty {
                        Text(message).font(.footnote).foregroundStyle(.secondary)
                    }
                }
                Section("Все приложения") {
                    Text("Чтобы убрать рекламу не только в Safari, но и в приложениях, установите DNS-профиль AdBlock-DNS.mobileconfig (см. README).")
                        .font(.footnote)
                    Button("Открыть настройки") {
                        if let url = URL(string: UIApplication.openSettingsURLString) {
                            UIApplication.shared.open(url)
                        }
                    }
                }
            }
            .navigationTitle("ZeroAds")
            .onAppear(perform: refreshState)
            .onReceive(NotificationCenter.default.publisher(
                for: UIApplication.willEnterForegroundNotification)) { _ in refreshState() }
        }
    }

    private func refreshState() {
        SFContentBlockerManager.getStateOfContentBlocker(withIdentifier: blockerIdentifier) { state, _ in
            DispatchQueue.main.async { isEnabled = state?.isEnabled ?? false }
        }
    }

    private func reload() {
        SFContentBlockerManager.reloadContentBlocker(withIdentifier: blockerIdentifier) { error in
            DispatchQueue.main.async {
                message = error.map { "Ошибка: \($0.localizedDescription)" } ?? "Правила обновлены ✓"
                refreshState()
            }
        }
    }
}
