import Foundation
import UniformTypeIdentifiers

/// Safari вызывает этот обработчик, чтобы получить список правил блокировки.
final class ContentBlockerRequestHandler: NSObject, NSExtensionRequestHandling {
    func beginRequest(with context: NSExtensionContext) {
        guard let url = Bundle.main.url(forResource: "blockerList", withExtension: "json"),
              let attachment = NSItemProvider(contentsOf: url) else {
            context.cancelRequest(withError: NSError(domain: "ZeroAds", code: 1))
            return
        }
        let item = NSExtensionItem()
        item.attachments = [attachment]
        context.completeRequest(returningItems: [item], completionHandler: nil)
    }
}
