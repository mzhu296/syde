import Foundation
import PDFKit
import AppKit

let input = URL(fileURLWithPath: CommandLine.arguments[1])
let output = URL(fileURLWithPath: CommandLine.arguments[2], isDirectory: true)
guard let doc = PDFDocument(url: input) else { fatalError("Could not open PDF") }
print("PAGES: \(doc.pageCount)")
for index in 0..<doc.pageCount {
    guard let page = doc.page(at: index) else { continue }
    print("\n===== PAGE \(index + 1) =====\n")
    print(page.string ?? "[no text]")

    let bounds = page.bounds(for: .mediaBox)
    let scale: CGFloat = 2.0
    let size = NSSize(width: bounds.width * scale, height: bounds.height * scale)
    let image = NSImage(size: size)
    image.lockFocus()
    NSColor.white.setFill()
    NSRect(origin: .zero, size: size).fill()
    guard let context = NSGraphicsContext.current?.cgContext else { fatalError("No context") }
    context.saveGState()
    context.scaleBy(x: scale, y: scale)
    page.draw(with: .mediaBox, to: context)
    context.restoreGState()
    image.unlockFocus()
    guard let tiff = image.tiffRepresentation,
          let rep = NSBitmapImageRep(data: tiff),
          let png = rep.representation(using: .png, properties: [:]) else { continue }
    try png.write(to: output.appendingPathComponent("page-\(index + 1).png"))
}
